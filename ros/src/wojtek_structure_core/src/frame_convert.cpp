/*
 * Structure Core -> ROS pixel and calibration conversions. See the header.
 */
#include "wojtek_structure_core/frame_convert.hpp"

#include <cmath>

namespace wojtek_structure_core
{

std::size_t depth_mm_float_to_u16(const float *src, std::size_t count,
                                  std::uint16_t min_mm, std::uint16_t max_mm,
                                  std::uint16_t *dst)
{
    if (src == nullptr || dst == nullptr)
        return 0;

    /* 0 is the encoding's "no return", so a valid pixel may not be 0. */
    const std::uint32_t lo = (min_mm == 0u) ? 1u : static_cast<std::uint32_t>(min_mm);
    const std::uint32_t hi = static_cast<std::uint32_t>(max_mm);

    std::size_t valid = 0;

    for (std::size_t i = 0; i < count; i++)
    {
        const float v = src[i];

        /* NaN and +/-inf are the SDK's "no return". The comparison below
         * would also reject NaN (every comparison with NaN is false), but
         * relying on that is the kind of cleverness that breaks under
         * -ffast-math, which a release build may well carry. */
        if (!std::isfinite(v))
        {
            dst[i] = 0u;
            continue;
        }

        /* Negative values cannot appear in a depth image, but lrintf of a
         * large negative float into an unsigned would wrap, so the range
         * test is done in a signed type wide enough to hold it. */
        const long mm = std::lrintf(v);
        if (mm < static_cast<long>(lo) || mm > static_cast<long>(hi))
        {
            dst[i] = 0u;
            continue;
        }

        dst[i] = static_cast<std::uint16_t>(mm);
        valid++;
    }

    return valid;
}

std::array<double, 9> camera_matrix(const Intrinsics &in)
{
    return {in.fx, 0.0,   in.cx,
            0.0,   in.fy, in.cy,
            0.0,   0.0,   1.0};
}

std::array<double, 12> projection_matrix(const Intrinsics &in)
{
    /* Monocular: no baseline, so the fourth column is zero. */
    return {in.fx, 0.0,   in.cx, 0.0,
            0.0,   in.fy, in.cy, 0.0,
            0.0,   0.0,   1.0,   0.0};
}

std::array<double, 5> distortion_plumb_bob(const Intrinsics &in)
{
    /* plumb_bob order, which interleaves the radial and tangential terms:
     * k1, k2, p1, p2, k3. */
    return {in.k1, in.k2, in.p1, in.p2, in.k3};
}

bool intrinsics_match(const Intrinsics &in, int width, int height)
{
    return in.width == width && in.height == height && in.fx > 0.0 && in.fy > 0.0;
}

void ClockSkew::update(double sdk_seconds, std::int64_t ros_nanos)
{
    const double offset_s = static_cast<double>(ros_nanos) * 1e-9 - sdk_seconds;

    if (n_ == 0)
    {
        first_offset_s_ = offset_s;
        drift_s_        = 0.0;
    }
    else
    {
        drift_s_ = offset_s - first_offset_s_;
        const double a = std::fabs(drift_s_);
        if (a > max_abs_drift_s_)
            max_abs_drift_s_ = a;
    }

    n_++;
}

}  // namespace wojtek_structure_core
