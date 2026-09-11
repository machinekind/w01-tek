/*
 * Occipital Structure Core (STO2D-C) ROS 2 driver node.
 *
 * Publishes the colour and depth streams the targeting chain needs, on
 * RealSense-shaped topic names so that anything written against the D435
 * moves over by changing a namespace:
 *
 *   <ns>/<name>/color/image_raw          rgb8
 *   <ns>/<name>/color/camera_info
 *   <ns>/<name>/depth/image_rect_raw     16UC1, millimetres, 0 = no return
 *   <ns>/<name>/depth/camera_info
 *   <ns>/<name>/infra/image_raw          mono16   (off by default)
 *   <ns>/<name>/status/healthy           Bool, latched
 *
 * Defaults put those at /targeting_camera/targeting_camera/..., which is the
 * namespace the targeting brief reserves and, more to the point, is NOT
 * /camera/camera/... -- that is the terrain D435, already live.
 *
 * WHAT THIS NODE IS NOT: it does no detection, no angle math and no
 * registration of depth into the colour frame. It is a driver.
 *
 * ---------------------------------------------------------------------------
 * SDK DEPENDENCY
 *
 * The Structure SDK (XRPro LLC, developer.structure.io) is closed source and
 * is NOT vendored into this repository -- it cannot be, the repository is
 * public and the SDK's licence is not ours to redistribute. Its location
 * comes in through STRUCTURE_SDK_DIR, declared at the top of CMakeLists.txt,
 * the same way every other out-of-tree secret or third-party path enters
 * this tree. Without it, CMake skips this node and still builds and tests
 * the conversion layer, so a workspace with no SDK (the RPi, CI) builds
 * clean rather than failing.
 *
 * Every SDK symbol this file touches is listed in the ENUM MAP and the
 * delegate below. They are written against the Structure SDK's documented
 * CaptureSession API; if your SDK revision spells one differently, the fix
 * is in one of those two places, not spread through the file.
 * ---------------------------------------------------------------------------
 */
#include <ST/CaptureSession.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "wojtek_structure_core/frame_convert.hpp"

namespace wojtek_structure_core
{

/* ======================================================== SDK ENUM MAP ====
 * The only place a header mismatch between SDK revisions can bite. Each
 * mapper takes the parameter string and returns the SDK enum, falling back
 * to a documented default with a warning rather than refusing to start --
 * a typo in a launch argument should not cost a bring-up attempt.
 */

static ST::StructureCoreDepthResolution depth_resolution_from(
    const std::string &s, rclcpp::Logger log)
{
    if (s == "QVGA" || s == "320x240")  return ST::StructureCoreDepthResolution::QVGA;
    if (s == "VGA"  || s == "640x480")  return ST::StructureCoreDepthResolution::VGA;
    if (s == "SXGA" || s == "1280x960") return ST::StructureCoreDepthResolution::SXGA;
    RCLCPP_WARN(log, "unknown depth_resolution '%s', using VGA", s.c_str());
    return ST::StructureCoreDepthResolution::VGA;
}

static ST::StructureCoreDepthRangeMode depth_range_mode_from(
    const std::string &s, rclcpp::Logger log)
{
    if (s == "VeryShort") return ST::StructureCoreDepthRangeMode::VeryShort;
    if (s == "Short")     return ST::StructureCoreDepthRangeMode::Short;
    if (s == "Medium")    return ST::StructureCoreDepthRangeMode::Medium;
    if (s == "Long")      return ST::StructureCoreDepthRangeMode::Long;
    if (s == "VeryLong")  return ST::StructureCoreDepthRangeMode::VeryLong;
    if (s == "Hybrid")    return ST::StructureCoreDepthRangeMode::Hybrid;
    RCLCPP_WARN(log, "unknown depth_range_mode '%s', using Medium", s.c_str());
    return ST::StructureCoreDepthRangeMode::Medium;
}

/* Copy the SDK's intrinsics into the SDK-free struct the conversion layer
 * speaks. Member-for-member on purpose: the field names match. */
static Intrinsics from_sdk(const ST::Intrinsics &i)
{
    Intrinsics out;
    out.width  = i.width;
    out.height = i.height;
    out.fx     = i.fx;
    out.fy     = i.fy;
    out.cx     = i.cx;
    out.cy     = i.cy;
    out.k1     = i.k1;
    out.k2     = i.k2;
    out.k3     = i.k3;
    out.p1     = i.p1;
    out.p2     = i.p2;
    return out;
}

/* ========================================================== the node ==== */

class StructureCoreNode : public rclcpp::Node
{
public:
    StructureCoreNode();
    ~StructureCoreNode() override;

    /* Called from the SDK's own thread. */
    void on_event(ST::CaptureSessionEventId event);
    void on_sample(const ST::CaptureSessionSample &sample);

private:
    void publish_depth(const ST::DepthFrame &frame, const rclcpp::Time &stamp);
    void publish_color(const ST::ColorFrame &frame, const rclcpp::Time &stamp);
    void publish_infrared(const ST::InfraredFrame &frame, const rclcpp::Time &stamp);
    void publish_camera_info(
        const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr &pub,
        const Intrinsics &in, const std::string &frame_id,
        const rclcpp::Time &stamp, bool distorted);
    void publish_static_tf();
    void set_healthy(bool healthy, const char *why);
    void watchdog();

    /* Parameters, read once at construction -- a camera that reconfigured
     * itself mid-stream would invalidate the CameraInfo a consumer has
     * already cached. Change one and restart. */
    std::string name_, ns_;
    double      depth_min_m_ = 0.3, depth_max_m_ = 5.0;
    bool        publish_depth_ = true, publish_color_ = true, publish_infra_ = false;
    double      watchdog_timeout_s_ = 2.0;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      depth_pub_, color_pub_, infra_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_, color_info_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr          healthy_pub_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster>       static_tf_;
    rclcpp::TimerBase::SharedPtr                               watchdog_timer_;

    /* Reused so a 30 Hz stream does not allocate a 600 kB buffer per frame
     * on a Pi. Guarded because the SDK delivers on its own thread. */
    std::mutex                 buf_mutex_;
    std::vector<std::uint16_t> depth_scratch_;

    std::atomic<std::uint64_t> frames_{0};
    std::atomic<std::int64_t>  last_frame_ns_{0};
    std::atomic<bool>          healthy_{false};
    std::atomic<bool>          streaming_{false};
    ClockSkew                  skew_;          /* buf_mutex_ guards this too */

    std::unique_ptr<ST::CaptureSession> session_;
    class Delegate;
    std::unique_ptr<Delegate> delegate_;
};

/* The SDK wants a delegate object; the node wants to be a node. Composition
 * rather than inheriting both. */
class StructureCoreNode::Delegate : public ST::CaptureSessionDelegate
{
public:
    explicit Delegate(StructureCoreNode *node) : node_(node) {}

    void captureSessionEventDidOccur(ST::CaptureSession *,
                                     ST::CaptureSessionEventId event) override
    {
        node_->on_event(event);
    }

    void captureSessionDidOutputSample(ST::CaptureSession *,
                                       const ST::CaptureSessionSample &sample) override
    {
        node_->on_sample(sample);
    }

private:
    StructureCoreNode *node_;
};

StructureCoreNode::StructureCoreNode() : rclcpp::Node("structure_core")
{
    /* ---- parameters ---- */
    name_ = declare_parameter<std::string>("camera_name", "targeting_camera");
    ns_   = declare_parameter<std::string>("camera_namespace", "targeting_camera");

    const auto serial = declare_parameter<std::string>("serial_number", "");
    const auto depth_res  = declare_parameter<std::string>("depth_resolution", "VGA");
    const auto range_mode = declare_parameter<std::string>("depth_range_mode", "Medium");
    const auto depth_fps  = declare_parameter<double>("depth_framerate", 30.0);
    const auto color_fps  = declare_parameter<double>("color_framerate", 30.0);

    publish_depth_ = declare_parameter<bool>("enable_depth", true);
    publish_color_ = declare_parameter<bool>("enable_color", true);
    publish_infra_ = declare_parameter<bool>("enable_infrared", false);

    depth_min_m_ = declare_parameter<double>("depth_min_m", 0.3);
    depth_max_m_ = declare_parameter<double>("depth_max_m", 5.0);

    /* Trades a frame of buffering for latency. On by default: this camera
     * feeds an aiming loop, where a stale frame is worse than a dropped
     * one. The terrain D435 would want the opposite. */
    const bool latency_reducer = declare_parameter<bool>("latency_reducer", true);

    /* The SDK's expensive depth correction. Off by default because it costs
     * CPU the Pi does not have spare with a second camera and a detector
     * running -- measure before turning it on. */
    const bool expensive_correction = declare_parameter<bool>("expensive_correction", false);

    watchdog_timeout_s_ = declare_parameter<double>("watchdog_timeout_s", 2.0);

    /* An inverted range makes every pixel invalid, which presents as "the
     * camera publishes nothing useful" -- say it here instead. */
    if (depth_min_m_ >= depth_max_m_)
        RCLCPP_ERROR(get_logger(),
                     "depth_min_m (%.3f) >= depth_max_m (%.3f): every pixel will be "
                     "published as no-return",
                     depth_min_m_, depth_max_m_);

    /* Placeholder, and it must stay loud: the colour and depth sensors sit a
     * baseline apart, so a bbox centre from the colour image does not index
     * the depth image at the same pixel. Either measure this and reproject,
     * or decide (as the brief still may) that this slice does not use depth
     * at all. A zero here is a lie that reads as a working system. */
    declare_parameter<double>("color_to_depth_baseline_m", 0.0);

    RCLCPP_INFO(get_logger(), "STO2D-C driver: ns=%s name=%s serial=%s",
                ns_.c_str(), name_.c_str(), serial.empty() ? "(first found)" : serial.c_str());

    /* ---- publishers ---- */
    const std::string base = "/" + ns_ + "/" + name_;
    const auto qos = rclcpp::SensorDataQoS();   /* best effort: a late frame is
                                                 * worthless to an aiming loop */

    if (publish_depth_)
    {
        depth_pub_      = create_publisher<sensor_msgs::msg::Image>(base + "/depth/image_rect_raw", qos);
        depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(base + "/depth/camera_info", qos);
    }
    if (publish_color_)
    {
        color_pub_      = create_publisher<sensor_msgs::msg::Image>(base + "/color/image_raw", qos);
        color_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(base + "/color/camera_info", qos);
    }
    if (publish_infra_)
        infra_pub_ = create_publisher<sensor_msgs::msg::Image>(base + "/infra/image_raw", qos);

    /* Latched: a consumer that starts late still learns the camera is down,
     * instead of waiting for a state change that already happened. */
    healthy_pub_ = create_publisher<std_msgs::msg::Bool>(
        base + "/status/healthy",
        rclcpp::QoS(1).reliable().transient_local());
    set_healthy(false, "not streaming yet");

    static_tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publish_static_tf();

    /* ---- session ---- */
    ST::CaptureSessionSettings settings;
    settings.source = ST::CaptureSessionSourceId::StructureCore;

    settings.structureCore.depthEnabled         = publish_depth_;
    settings.structureCore.visibleEnabled       = publish_color_;
    settings.structureCore.infraredEnabled      = publish_infra_;
    /* The brief cuts the IMU: it only buys aim compensation while walking,
     * and the fallback plan is stop-then-lock-then-track. Not "off for now"
     * -- off by decision. */
    settings.structureCore.accelerometerEnabled = false;
    settings.structureCore.gyroscopeEnabled     = false;

    settings.structureCore.depthResolution  = depth_resolution_from(depth_res, get_logger());
    settings.structureCore.depthRangeMode    = depth_range_mode_from(range_mode, get_logger());
    settings.structureCore.depthFramerate    = static_cast<float>(depth_fps);
    settings.structureCore.visibleFramerate  = static_cast<float>(color_fps);
    settings.structureCore.latencyReducerEnabled = latency_reducer;
    settings.applyExpensiveCorrection            = expensive_correction;

    /* Pin the device when a serial is given. With two depth cameras on one
     * Pi, "whichever enumerated first" is a coin flip at every boot. */
    if (!serial.empty())
        settings.structureCore.sensorSerial = serial.c_str();

    delegate_ = std::make_unique<Delegate>(this);
    session_  = std::make_unique<ST::CaptureSession>();
    session_->setDelegate(delegate_.get());

    if (!session_->startMonitoring(settings))
    {
        /* Hard failure: no device, no permission (udev), or a settings
         * combination the sensor refuses. Say all three, because the SDK
         * does not distinguish them here. */
        RCLCPP_FATAL(get_logger(),
                     "startMonitoring failed. Check: the sensor is plugged into USB3, "
                     "udev rules from the SDK are installed (otherwise only root sees it), "
                     "and depth_resolution/depth_framerate is a combination this unit "
                     "supports.");
        throw std::runtime_error("ST::CaptureSession::startMonitoring failed");
    }

    watchdog_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() { watchdog(); });

    RCLCPP_INFO(get_logger(), "monitoring; waiting for the sensor to report Ready");
}

StructureCoreNode::~StructureCoreNode()
{
    if (session_ && streaming_.load())
        session_->stopStreaming();
}

/* ----------------------------------------------------------- SDK thread -- */

void StructureCoreNode::on_event(ST::CaptureSessionEventId event)
{
    switch (event)
    {
    case ST::CaptureSessionEventId::Booting:
        RCLCPP_INFO(get_logger(), "sensor booting");
        break;

    case ST::CaptureSessionEventId::Detected:
        RCLCPP_INFO(get_logger(), "sensor detected");
        break;

    case ST::CaptureSessionEventId::Connected:
        RCLCPP_INFO(get_logger(), "sensor connected");
        break;

    case ST::CaptureSessionEventId::Ready:
        /* Streaming may only start once the sensor says Ready. Starting
         * earlier is the documented way to get a session that reports
         * success and never delivers a frame. */
        RCLCPP_INFO(get_logger(), "sensor ready; starting streams");
        if (session_->startStreaming())
            streaming_.store(true);
        else
            RCLCPP_ERROR(get_logger(), "startStreaming failed");
        break;

    case ST::CaptureSessionEventId::Disconnected:
        streaming_.store(false);
        set_healthy(false, "sensor disconnected");
        /* Deliberately not re-enumerating from here: this callback runs on
         * the SDK's thread, and tearing down the session that owns it from
         * inside it is how you get a deadlock on a USB hiccup. The session
         * keeps monitoring and will re-report Ready when the device comes
         * back; the watchdog reports the gap in the meantime. */
        RCLCPP_ERROR(get_logger(), "sensor disconnected (USB). Monitoring continues.");
        break;

    case ST::CaptureSessionEventId::Error:
        streaming_.store(false);
        set_healthy(false, "sensor error");
        RCLCPP_ERROR(get_logger(), "sensor reported an error");
        break;

    default:
        RCLCPP_DEBUG(get_logger(), "unhandled capture session event %d",
                     static_cast<int>(event));
        break;
    }
}

void StructureCoreNode::on_sample(const ST::CaptureSessionSample &sample)
{
    /* Stamped here, on arrival, with the ROS clock -- not with the frame's
     * own SDK timestamp. The two clocks have different epochs, and this
     * workspace has already paid for mixing them: ground truth stamped from
     * the controller manager's monotonic clock produced a TF tree in which
     * every frame was in the buffer and none could be looked up. The SDK
     * stamp is tracked separately, below, because its drift against this one
     * is the only thing that tells a slow sensor from a slow node. */
    const rclcpp::Time stamp = now();

    switch (sample.type)
    {
    case ST::CaptureSessionSample::Type::SynchronizedFrames:
        if (publish_depth_ && sample.depthFrame.isValid())
            publish_depth(sample.depthFrame, stamp);
        if (publish_color_ && sample.visibleFrame.isValid())
            publish_color(sample.visibleFrame, stamp);
        if (publish_infra_ && sample.infraredFrame.isValid())
            publish_infrared(sample.infraredFrame, stamp);
        break;

    case ST::CaptureSessionSample::Type::DepthFrame:
        if (publish_depth_ && sample.depthFrame.isValid())
            publish_depth(sample.depthFrame, stamp);
        break;

    case ST::CaptureSessionSample::Type::VisibleFrame:
        if (publish_color_ && sample.visibleFrame.isValid())
            publish_color(sample.visibleFrame, stamp);
        break;

    case ST::CaptureSessionSample::Type::InfraredFrame:
        if (publish_infra_ && sample.infraredFrame.isValid())
            publish_infrared(sample.infraredFrame, stamp);
        break;

    default:
        return;   /* IMU and the rest: not subscribed, not an error */
    }

    frames_.fetch_add(1);
    last_frame_ns_.store(stamp.nanoseconds());
    if (!healthy_.load())
        set_healthy(true, "frames arriving");
}

void StructureCoreNode::publish_depth(const ST::DepthFrame &frame,
                                      const rclcpp::Time &stamp)
{
    const int w = frame.width(), h = frame.height();
    if (w <= 0 || h <= 0)
        return;

    const float *mm = frame.depthInMillimeters();
    if (mm == nullptr)
        return;

    const auto count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

    /* Clamped into the encoding before narrowing: a depth_max_m of 70 would
     * otherwise wrap through uint16 into a 4.5 m limit, which is a plausible
     * number and therefore the worst kind of wrong. */
    const auto to_mm = [](double m) -> std::uint16_t {
        const double mm = m * 1000.0;
        if (mm < 1.0)     return 1u;
        if (mm > 65535.0) return 65535u;
        return static_cast<std::uint16_t>(mm);
    };
    const std::uint16_t min_mm = to_mm(depth_min_m_);
    const std::uint16_t max_mm = to_mm(depth_max_m_);

    std::size_t valid = 0;
    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    {
        std::lock_guard<std::mutex> lock(buf_mutex_);
        depth_scratch_.resize(count);
        valid = depth_mm_float_to_u16(mm, count, min_mm, max_mm, depth_scratch_.data());

        msg->data.resize(count * sizeof(std::uint16_t));
        std::memcpy(msg->data.data(), depth_scratch_.data(), msg->data.size());

        skew_.update(frame.timestamp(), stamp.nanoseconds());
        if (skew_.ready() && skew_.max_abs_drift_s() > 0.25)
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "sensor/ROS clock drift %.3f s (peak %.3f s): frames are "
                                 "queueing somewhere",
                                 skew_.drift_s(), skew_.max_abs_drift_s());
    }

    msg->header.stamp    = stamp;
    msg->header.frame_id = name_ + "_depth_optical_frame";
    msg->height          = static_cast<std::uint32_t>(h);
    msg->width           = static_cast<std::uint32_t>(w);
    msg->encoding        = "16UC1";
    msg->is_bigendian    = 0;
    msg->step            = static_cast<std::uint32_t>(w * sizeof(std::uint16_t));

    /* Streaming with nothing in range looks exactly like a broken driver.
     * Say which one it is. */
    if (valid == 0)
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "depth frame has no valid pixels: everything outside "
                             "[%.2f, %.2f] m. Wrong depth_range_mode, or nothing in view.",
                             depth_min_m_, depth_max_m_);

    depth_pub_->publish(std::move(msg));
    publish_camera_info(depth_info_pub_, from_sdk(frame.intrinsics()),
                        name_ + "_depth_optical_frame", stamp, /*distorted=*/false);
}

void StructureCoreNode::publish_color(const ST::ColorFrame &frame,
                                      const rclcpp::Time &stamp)
{
    const int w = frame.width(), h = frame.height();
    if (w <= 0 || h <= 0)
        return;

    const std::uint8_t *rgb = frame.rgbData();
    if (rgb == nullptr)
    {
        /* The mono STO2D has no colour sensor. The -C in STO2D-C is the
         * colour one, so a null here on that part number means the visible
         * stream was never enabled, not that the hardware lacks it. */
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "visible frame carries no RGB data");
        return;
    }

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp    = stamp;
    msg->header.frame_id = name_ + "_color_optical_frame";
    msg->height          = static_cast<std::uint32_t>(h);
    msg->width           = static_cast<std::uint32_t>(w);
    msg->encoding        = "rgb8";
    msg->is_bigendian    = 0;
    msg->step            = static_cast<std::uint32_t>(w * 3);
    msg->data.resize(static_cast<std::size_t>(w) * h * 3);
    std::memcpy(msg->data.data(), rgb, msg->data.size());

    color_pub_->publish(std::move(msg));
    publish_camera_info(color_info_pub_, from_sdk(frame.intrinsics()),
                        name_ + "_color_optical_frame", stamp, /*distorted=*/true);
}

void StructureCoreNode::publish_infrared(const ST::InfraredFrame &frame,
                                         const rclcpp::Time &stamp)
{
    const int w = frame.width(), h = frame.height();
    if (w <= 0 || h <= 0 || frame.data() == nullptr)
        return;

    auto msg = std::make_unique<sensor_msgs::msg::Image>();
    msg->header.stamp    = stamp;
    msg->header.frame_id = name_ + "_infra_optical_frame";
    msg->height          = static_cast<std::uint32_t>(h);
    msg->width           = static_cast<std::uint32_t>(w);
    msg->encoding        = "mono16";
    msg->is_bigendian    = 0;
    msg->step            = static_cast<std::uint32_t>(w * sizeof(std::uint16_t));
    msg->data.resize(static_cast<std::size_t>(w) * h * sizeof(std::uint16_t));
    std::memcpy(msg->data.data(), frame.data(), msg->data.size());

    infra_pub_->publish(std::move(msg));
}

void StructureCoreNode::publish_camera_info(
    const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr &pub,
    const Intrinsics &in, const std::string &frame_id,
    const rclcpp::Time &stamp, bool distorted)
{
    if (!pub)
        return;

    sensor_msgs::msg::CameraInfo info;
    info.header.stamp    = stamp;
    info.header.frame_id = frame_id;
    info.width           = static_cast<std::uint32_t>(in.width);
    info.height          = static_cast<std::uint32_t>(in.height);

    const auto K = camera_matrix(in);
    const auto P = projection_matrix(in);
    std::copy(K.begin(), K.end(), info.k.begin());
    std::copy(P.begin(), P.end(), info.p.begin());
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    info.distortion_model = "plumb_bob";
    if (distorted)
    {
        const auto D = distortion_plumb_bob(in);
        info.d.assign(D.begin(), D.end());
    }
    else
    {
        /* Depth comes out of the SDK already rectified. Publishing the
         * visible camera's coefficients on it would make a consumer
         * undistort twice. */
        info.d.assign(5, 0.0);
    }

    pub->publish(info);
}

void StructureCoreNode::publish_static_tf()
{
    /* camera_link (REP-103: x forward, y left, z up) -> optical frames
     * (REP-145: z forward, x right, y down). The rotation is rpy
     * (-pi/2, 0, -pi/2), i.e. the quaternion below -- the same edge the
     * RealSense driver publishes, so a consumer's frame maths carries over.
     *
     * camera_link -> base_link is NOT published here: that is the mount, it
     * is a singleton, and it belongs to whatever bringup owns the robot --
     * exactly as wojtek_perception_bringup keeps its extrinsics opt-out.
     * Measure it before trusting anything: 1 deg of pitch error is 52 mm at
     * 3 m, larger than this class of sensor's own noise there. */
    const double qx = -0.5, qy = 0.5, qz = -0.5, qw = 0.5;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    const auto stamp = now();

    for (const char *suffix : {"_depth_optical_frame", "_color_optical_frame",
                               "_infra_optical_frame"})
    {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp            = stamp;
        t.header.frame_id         = name_ + "_link";
        t.child_frame_id          = name_ + suffix;
        t.transform.rotation.x    = qx;
        t.transform.rotation.y    = qy;
        t.transform.rotation.z    = qz;
        t.transform.rotation.w    = qw;
        tfs.push_back(t);
    }

    /* The colour sensor sits a baseline from the depth sensor. The default
     * of 0 is a placeholder, and it is the reason a bbox centre cannot be
     * used to index the depth image until someone measures it (or reads it
     * out of the SDK's own extrinsics). */
    const double baseline = get_parameter("color_to_depth_baseline_m").as_double();
    if (baseline != 0.0)
        tfs[1].transform.translation.x = baseline;

    static_tf_->sendTransform(tfs);
}

void StructureCoreNode::set_healthy(bool healthy, const char *why)
{
    healthy_.store(healthy);
    std_msgs::msg::Bool msg;
    msg.data = healthy;
    healthy_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "healthy=%s (%s)", healthy ? "true" : "false", why);
}

void StructureCoreNode::watchdog()
{
    const auto last = last_frame_ns_.load();
    if (last == 0)
        return;   /* nothing has arrived yet; the Ready path logs that */

    const double age_s = (now().nanoseconds() - last) * 1e-9;
    if (age_s > watchdog_timeout_s_ && healthy_.load())
        set_healthy(false, "no frames");
}

}  // namespace wojtek_structure_core

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try
    {
        rclcpp::spin(std::make_shared<wojtek_structure_core::StructureCoreNode>());
    }
    catch (const std::exception &e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("structure_core"), "fatal: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
