/*
 * Host tests for the Structure Core conversion layer.
 *
 * No SDK, no ROS, no camera: `g++ test_frame_convert.cpp frame_convert.cpp`
 * and run it. colcon runs the same binary through add_test.
 */
#include "wojtek_structure_core/frame_convert.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace wojtek_structure_core;

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            failures++;                                                     \
        }                                                                   \
    } while (0)

/* ------------------------------------------------------------------ depth */

static void test_nan_and_inf_become_no_return(void)
{
    const float src[] = {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         1000.0f};
    std::uint16_t dst[4] = {9, 9, 9, 9};

    const std::size_t valid = depth_mm_float_to_u16(src, 4, 300, 5000, dst);

    CHECK(valid == 1);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 0);
    CHECK(dst[2] == 0);
    CHECK(dst[3] == 1000);
}

static void test_out_of_range_is_dropped_not_clamped(void)
{
    /* A clamped pixel is indistinguishable from a real surface at the range
     * limit, and the gimbal would aim at it. */
    const float   src[] = {299.0f, 300.0f, 5000.0f, 5001.0f};
    std::uint16_t dst[4];

    const std::size_t valid = depth_mm_float_to_u16(src, 4, 300, 5000, dst);

    CHECK(valid == 2);
    CHECK(dst[0] == 0);      /* below min */
    CHECK(dst[1] == 300);    /* min is inclusive */
    CHECK(dst[2] == 5000);   /* max is inclusive */
    CHECK(dst[3] == 0);      /* above max */
}

static void test_zero_min_cannot_alias_no_return(void)
{
    /* With min_mm = 0 a genuine 0 mm reading would encode as "no return",
     * so the floor is raised to 1. */
    const float   src[] = {0.0f, 0.4f, 1.0f};
    std::uint16_t dst[3];

    const std::size_t valid = depth_mm_float_to_u16(src, 3, 0, 5000, dst);

    CHECK(valid == 1);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 0);   /* rounds to 0, which is not a valid value */
    CHECK(dst[2] == 1);
}

static void test_rounding_is_nearest(void)
{
    const float   src[] = {999.4f, 999.5f, 1000.6f};
    std::uint16_t dst[3];

    depth_mm_float_to_u16(src, 3, 1, 65535, dst);

    CHECK(dst[0] == 999);
    CHECK(dst[1] == 1000);
    CHECK(dst[2] == 1001);
}

static void test_negative_does_not_wrap(void)
{
    /* A negative float cast straight to uint16 would wrap to a plausible
     * short range. It must read as no-return instead. */
    const float   src[] = {-1.0f, -70000.0f};
    std::uint16_t dst[2];

    const std::size_t valid = depth_mm_float_to_u16(src, 2, 1, 65535, dst);

    CHECK(valid == 0);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 0);
}

static void test_beyond_uint16_does_not_wrap(void)
{
    /* 70000 mm is inside Long range mode's ambition and outside uint16. */
    const float   src[] = {70000.0f, 65535.0f};
    std::uint16_t dst[2];

    const std::size_t valid = depth_mm_float_to_u16(src, 2, 1, 65535, dst);

    CHECK(valid == 1);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 65535);
}

static void test_null_and_empty_are_safe(void)
{
    std::uint16_t dst[1] = {7};
    const float   src[1] = {1.0f};

    CHECK(depth_mm_float_to_u16(nullptr, 1, 1, 5000, dst) == 0);
    CHECK(depth_mm_float_to_u16(src, 1, 1, 5000, nullptr) == 0);
    CHECK(depth_mm_float_to_u16(src, 0, 1, 5000, dst) == 0);
    CHECK(dst[0] == 7);   /* untouched */
}

static void test_all_invalid_frame_reports_zero_valid(void)
{
    /* The failure that looks exactly like a broken driver: frames arriving,
     * nothing in range. */
    std::vector<float>         src(640 * 480, std::numeric_limits<float>::quiet_NaN());
    std::vector<std::uint16_t> dst(640 * 480, 1);

    const std::size_t valid =
        depth_mm_float_to_u16(src.data(), src.size(), 300, 5000, dst.data());

    CHECK(valid == 0);
    CHECK(dst[0] == 0);
    CHECK(dst[src.size() - 1] == 0);
}

/* ------------------------------------------------------------ calibration */

static Intrinsics sample_intrinsics(void)
{
    Intrinsics in;
    in.width  = 640;
    in.height = 480;
    in.fx     = 570.5;
    in.fy     = 571.0;
    in.cx     = 319.5;
    in.cy     = 239.5;
    in.k1     = 0.1;
    in.k2     = -0.2;
    in.k3     = 0.3;
    in.p1     = 0.01;
    in.p2     = 0.02;
    return in;
}

static void test_camera_matrix_layout(void)
{
    const auto K = camera_matrix(sample_intrinsics());

    CHECK(K[0] == 570.5);
    CHECK(K[1] == 0.0);
    CHECK(K[2] == 319.5);
    CHECK(K[4] == 571.0);
    CHECK(K[5] == 239.5);
    CHECK(K[8] == 1.0);
}

static void test_projection_has_no_baseline(void)
{
    const auto P = projection_matrix(sample_intrinsics());

    CHECK(P[0] == 570.5);
    CHECK(P[2] == 319.5);
    CHECK(P[3] == 0.0);    /* Tx */
    CHECK(P[7] == 0.0);    /* Ty */
    CHECK(P[10] == 1.0);
    CHECK(P[11] == 0.0);
}

static void test_distortion_is_plumb_bob_order(void)
{
    /* k1, k2, p1, p2, k3 -- NOT the SDK struct's declaration order. */
    const auto D = distortion_plumb_bob(sample_intrinsics());

    CHECK(D[0] == 0.1);    /* k1 */
    CHECK(D[1] == -0.2);   /* k2 */
    CHECK(D[2] == 0.01);   /* p1 */
    CHECK(D[3] == 0.02);   /* p2 */
    CHECK(D[4] == 0.3);    /* k3 */
}

static void test_intrinsics_match_guards_resolution_change(void)
{
    const Intrinsics in = sample_intrinsics();

    CHECK(intrinsics_match(in, 640, 480));
    CHECK(!intrinsics_match(in, 1280, 960));

    Intrinsics empty;
    CHECK(!intrinsics_match(empty, 0, 0));   /* fx = 0 is not a calibration */
}

/* ------------------------------------------------------------- clock skew */

static void test_clock_skew_is_relative_to_first_sample(void)
{
    ClockSkew s;

    CHECK(!s.ready());

    /* SDK clock at 100 s, ROS clock at 1000 s: a 900 s epoch offset, which
     * is exactly the thing that must NOT be reported as drift. */
    s.update(100.0, 1000000000000LL);
    CHECK(s.ready() == false);       /* one sample establishes the baseline */
    CHECK(s.drift_s() == 0.0);

    s.update(100.1, 1000100000000LL);   /* both advanced 0.1 s: no drift */
    CHECK(s.ready());
    CHECK(std::fabs(s.drift_s()) < 1e-6);

    s.update(100.2, 1000250000000LL);   /* ROS advanced 50 ms more */
    CHECK(std::fabs(s.drift_s() - 0.05) < 1e-6);
    CHECK(std::fabs(s.max_abs_drift_s() - 0.05) < 1e-6);

    s.update(100.3, 1000300000000LL);   /* back in step: peak is retained */
    CHECK(std::fabs(s.drift_s()) < 1e-6);
    CHECK(std::fabs(s.max_abs_drift_s() - 0.05) < 1e-6);
    CHECK(s.samples() == 4);
}

int main(void)
{
    test_nan_and_inf_become_no_return();
    test_out_of_range_is_dropped_not_clamped();
    test_zero_min_cannot_alias_no_return();
    test_rounding_is_nearest();
    test_negative_does_not_wrap();
    test_beyond_uint16_does_not_wrap();
    test_null_and_empty_are_safe();
    test_all_invalid_frame_reports_zero_valid();

    test_camera_matrix_layout();
    test_projection_has_no_baseline();
    test_distortion_is_plumb_bob_order();
    test_intrinsics_match_guards_resolution_change();

    test_clock_skew_is_relative_to_first_sample();

    if (failures != 0)
    {
        std::printf("test_frame_convert: %d FAILED\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("test_frame_convert: all passed\n");
    return EXIT_SUCCESS;
}
