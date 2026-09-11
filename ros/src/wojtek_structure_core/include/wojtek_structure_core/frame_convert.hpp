/*
 * Structure Core -> ROS pixel and calibration conversions.
 *
 * No SDK, no ROS, no threads: everything that can be wrong about a pixel
 * format or a calibration matrix is reachable from a host test, leaving the
 * SDK layer (src/structure_core_node.cpp) with USB, delegate threads and
 * reconnects. That is the same split ms5611_math.c / ms5611.c uses in the
 * flight firmware, and the same reason wojtek_policy's numpy runtime has no
 * ROS imports.
 *
 * The conversions here exist because the Structure SDK and this workspace
 * disagree about how a depth image is spelled:
 *
 *   Structure SDK   float millimetres, NaN for "no return"
 *   this workspace  16UC1 millimetres, 0 for "no return"  (RealSense's
 *                   convention, which cloud_reduce and the D435 config
 *                   already assume)
 *
 * Matching RealSense rather than passing the SDK's float through is what
 * lets anything already written against the D435 depth topics work against
 * a Structure Core by changing a topic name.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace wojtek_structure_core
{

/* The subset of ST::Intrinsics this driver uses, copied out of the SDK type
 * at the boundary so this header stays SDK-free. Field names match the SDK's
 * so the adapter in the node is a plain member-for-member copy. */
struct Intrinsics
{
    int    width  = 0;
    int    height = 0;
    double fx     = 0.0;
    double fy     = 0.0;
    double cx     = 0.0;
    double cy     = 0.0;
    /* Brown-Conrady. The SDK reports these for the visible camera; depth
     * comes already undistorted, so a depth CameraInfo carries zeros. */
    double k1 = 0.0, k2 = 0.0, k3 = 0.0, p1 = 0.0, p2 = 0.0;
};

/* Depth conversion. Writes count pixels into dst and returns how many of
 * them are valid (non-zero).
 *
 * A pixel becomes 0 when it is not finite (the SDK's "no return"), or when
 * it falls outside [min_mm, max_mm] inclusive. Out-of-range is dropped
 * rather than clamped on purpose: a clamped pixel is indistinguishable from
 * a real surface at the limit, and a planner or a gimbal would aim at it.
 *
 * min_mm is raised to 1 if it is 0, because 0 is the encoding's own "no
 * return" and a valid pixel may not collide with it.
 *
 * The returned count is the cheapest available diagnostic for the failure
 * that looks exactly like a broken driver: a camera that streams perfectly
 * while every pixel is out of range (sensor pointed at a far wall, range
 * mode too short, IR projector dead). Zero valid pixels with rising frame
 * counts is that, not a dropped link.
 */
std::size_t depth_mm_float_to_u16(const float *src, std::size_t count,
                                  std::uint16_t min_mm, std::uint16_t max_mm,
                                  std::uint16_t *dst);

/* CameraInfo fields, in ROS order.
 *
 * K is row-major 3x3. P is row-major 3x4 with the fourth column zero: these
 * are monocular streams, so there is no baseline term. D is plumb_bob's
 * [k1, k2, p1, p2, k3] -- note that order, it is not the order the SDK
 * struct declares them in, and getting it wrong quietly warps every
 * undistort downstream. */
std::array<double, 9>  camera_matrix(const Intrinsics &in);
std::array<double, 12> projection_matrix(const Intrinsics &in);
std::array<double, 5>  distortion_plumb_bob(const Intrinsics &in);

/* True when the intrinsics describe an image of exactly this size. The SDK
 * reports intrinsics per frame, and a resolution change mid-session would
 * otherwise publish a CameraInfo that does not match its image. */
bool intrinsics_match(const Intrinsics &in, int width, int height);

/* SDK clock vs ROS clock.
 *
 * Frames are stamped with the ROS clock at arrival, not with the SDK's
 * timestamp: the two run on different epochs, and this workspace has already
 * paid for mixing them once -- ground truth stamped with the controller
 * manager's monotonic clock produced a TF tree where every frame was present
 * in the buffer and none could be looked up. Arrival stamping costs one
 * transport hop of accuracy and cannot produce that failure.
 *
 * The SDK timestamp is still worth watching, because the delta between the
 * two clocks is the only thing that distinguishes "the sensor is slow" from
 * "this node is slow". This tracks that delta's drift relative to the first
 * sample, in seconds.
 */
class ClockSkew
{
public:
    void   update(double sdk_seconds, std::int64_t ros_nanos);
    bool   ready() const { return n_ > 1; }
    double drift_s() const { return drift_s_; }      /* latest - first */
    double max_abs_drift_s() const { return max_abs_drift_s_; }
    std::size_t samples() const { return n_; }

private:
    std::size_t n_                = 0;
    double      first_offset_s_   = 0.0;
    double      drift_s_          = 0.0;
    double      max_abs_drift_s_  = 0.0;
};

}  // namespace wojtek_structure_core
