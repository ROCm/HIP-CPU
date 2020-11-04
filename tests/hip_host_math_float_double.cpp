/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <algorithm>
#include <iterator>

using namespace std;

TEST_CASE("(host) erfcinvf(float)", "[host][math][erfcinvf]")
{
    float Val[]{0.1f, 1.2f, 1.f, 0.9f};
    float Out[]{1.16309f, -0.179144f, 0.f, 0.088856f};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfcinvf(x) == Approx{y};
    }));
}

TEST_CASE("(host) erfcxf(float)", "[host][math][erfcxf]")
{
    float Val[]{-0.5f, 9.f, 3.2f, 1.f};
    float Out[]{1.95236f, 0.06231f, 0.16873f, 0.427584f};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfcxf(x) == Approx{y}.scale(1.0);
    }));
}

TEST_CASE("(host) erfinvf(float)", "[host][math][erfinvf]")
{
    float Val[]{0.f, -0.5f, 0.9f, -0.2f};
    float Out[]{0.f, -0.476936f, 1.1631f, -0.179143f};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfinvf(x) == Approx{y};
    }));
}

TEST_CASE("(host) fdividef(float)", "[host][math][fdividef]")
{
    float Val[]{0.f, -0.5f, 0.9f, -0.2f};
    float Out[]{1.f, -0.4769f, 1.1631f, -0.1791f};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return fdividef(x, y) == Approx{x / y};
    }));
}

TEST_CASE("(host) erfcinv(double)", "[host][math][erfcinv]")
{
    double Val[]{0.1, 1.2, 1, 0.9};
    double Out[]{1.16309, -0.179144, 0, 0.0888559889};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfcinv(x) == Approx{y};
    }));
}

TEST_CASE("(host) erfcx(double)", "[host][math][erfcx]")
{
    double Val[]{-0.5, 15, 3.2, 1};
    double Out[]{1.9523604892, 0.0375296064, 0.1687280968, 0.4275835762};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfcx(x) == Approx{y};
    }));
}

TEST_CASE("(host) erfinv(double)", "[host][math][erfinv]")
{
    double Val[]{0, -0.5, 0.9, -0.2};
    double Out[]{0, -0.476936287, 1.1631, -0.1791434596};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return erfinv(x) == Approx{y};
    }));
}

TEST_CASE("(host) fdivide(double)", "[host][math][fdivide]")
{
    double Val[]{0, -0.5, 0.9, -0.2};
    double Out[]{1, -0.4769, 1.1631, -0.1791};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return fdivide(x, y) == Approx{x / y};
    }));
}

TEST_CASE("(host) modff(float)", "[host][math][modff]")
{
    float Val[]{0.f, -0.5f, 0.9f, -0.2f};
    float iPtr[]{0, 0, 0, 0};
    float frac[]{0, -0.5f, 0.9f, -0.2f};
    float Out[]{1, 1, 1, 1};

    for (auto i = 0u; i != size(Val); ++i) {
        REQUIRE(frac[i] == Approx{modff(Val[i], Out + i)});
        REQUIRE(iPtr[i] == Approx{Out[i]});
    }
}

TEST_CASE("(host) modf(double)", "[host][math][modf]")
{
    double Val[]{0, -0.5, 0.9, -0.2};
    double iPtr[]{0, 0, 0, 0};
    double frac[]{0, -0.5, 0.9, -0.2};
    double Out[]{1, 1, 1, 1};

    for (auto i = 0u; i != size(Val); ++i) {
        REQUIRE(frac[i] == Approx{modf(Val[i], Out + i)});
        REQUIRE(iPtr[i] == Approx{Out[i]});
    }
}

TEST_CASE("(host) nextafterf(float)", "[host][math][nextafterf]")
{
    float Val[]{0, -0.5f, 0.9f, -0.2f};

    REQUIRE(all_of(cbegin(Val), cend(Val), [](auto&& x) {
        return nextafterf(x, 1) == Approx{x}.scale(1.0);
    }));
}

TEST_CASE("(host) nextafter(double)", "[host][math][nextafter]")
{
    double Val[]{0, -0.5, 0.9, -0.2};

    REQUIRE(all_of(cbegin(Val), cend(Val), [](auto&& x) {
        return nextafter(x, 1) == Approx{x}.scale(1.0);
    }));
}
TEST_CASE("(host) norm3df(float)", "[host][math][norm3df]")
{
    REQUIRE(
        norm3df(0.f, 1.f, 2.f) ==
        Approx{std::sqrt(0.f * 0.f + 1.f * 1.f + 2.f * 2.f)});
}

TEST_CASE("(host) norm3d(double)", "[host][math][norm3d]")
{
    REQUIRE(
        norm3d(0., 1., 2.) == Approx{std::sqrt(0. * 0. + 1. * 1. + 2. * 2.)});
}

TEST_CASE("(host) norm4df(float)", "[host][math][norm4df]")
{
    REQUIRE(
        norm4df(0.f, 1.f, 2.f, 3.f) ==
        Approx{std::sqrt(0.f * 0.f + 1.f * 1.f + 2.f * 2.f + 3.f * 3.f)});
}

TEST_CASE("(host) norm4d(double)", "[host][math][norm4d]")
{
    REQUIRE(
        norm4d(0., 1., 2., 3.) ==
        Approx{std::sqrt(0. * 0. + 1. * 1. + 2. * 2. + 3. * 3.)});
}

TEST_CASE("(host) normcdff(float)", "[host][math][normcdff]")
{
    float Val[]{0, 1};
    float Out[]{0.5f, 0.841345f};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return normcdff(x) == Approx{y};
    }));
}

TEST_CASE("(host) normcdf(double)", "[host][math][normcdf]")
{
    double Val[]{0, 1};
    double Out[]{0.5, 0.841345};

    REQUIRE(equal(cbegin(Val), cend(Val), cbegin(Out), [](auto&& x, auto&& y) {
        return normcdf(x) == Approx{y};
    }));
}

TEST_CASE("(host) normcdfinvf(float)", "[host][math][normcdfinvf]")
{
    float Val[]{0.5f, 0.8413f};

    REQUIRE(all_of(cbegin(Val), cend(Val), [](auto&& x) {
        return normcdfinvf(normcdff(x)) == Approx{x};
    }));
}

TEST_CASE("(host) normcdfinv(double)", "[host][math][normcdfinv]")
{
    double Val[]{0.5, 0.8413};

    REQUIRE(all_of(cbegin(Val), cend(Val), [](auto&& x) {
        return normcdfinv(normcdf(x)) == Approx{x};
    }));
}

TEST_CASE("(host) rcbrtf(float)", "[host][math][rcbrtf]")
{
    REQUIRE(rcbrtf(1.f) == Approx{1.f});
}

TEST_CASE("(host) rcbrt(double)", "[host][math][rcbrt]")
{
    REQUIRE(rcbrt(1.) == Approx{1.});
}

TEST_CASE("(host) rhypotf(float)", "[host][math][rhypotf]")
{
    REQUIRE(
        rhypotf(1.f, 2.f) == Approx{1.f / std::sqrt(1.f * 1.f + 2.f * 2.f)});
}

TEST_CASE("(host) rhypot(double)", "[host][math][rhypot]")
{
    REQUIRE(rhypotf(1., 2.) == Approx{1. / std::sqrt(1. * 1. + 2. * 2.)});
}

TEST_CASE("(host) rnorm3df(float)", "[host][math][rnorm3df]")
{
    REQUIRE(
        rnorm3df(0.f, 1.f, 2.f) ==
        Approx{1.f / std::sqrt(0.f * 0.f + 1.f * 1.f + 2.f * 2.f)});
}

TEST_CASE("(host) rnorm3d(double)", "[host][math][rnorm3d]")
{
    REQUIRE(
        rnorm3df(0., 1., 2.) ==
        Approx{1. / std::sqrt(0. * 0. + 1. * 1. + 2. * 2.)});
}

TEST_CASE("(host) rnorm4df(float)", "[host][math][rnorm4d]")
{
    REQUIRE(
        rnorm4df(0.f, 1.f, 2.f, 3.f) ==
        Approx{1.f / std::sqrt(0.f * 0.f + 1.f * 1.f + 2.f * 2.f + 3.f * 3.f)});
}

TEST_CASE("(host) rnorm4d(double)", "[host][math][rnorm4d]")
{
    REQUIRE(
        rnorm4d(0., 1., 2., 3.) ==
        Approx{1. / std::sqrt(0. * 0. + 1. * 1. + 2. * 2. + 3. * 3.)});
}

TEST_CASE("(host) rnormf(float)", "[host][math][rnormf]")
{
    float A[]{0, 1, 2, 3};

    REQUIRE(rnormf(3, A) == Approx{rnorm3df(A[0], A[1], A[2])});
    REQUIRE(rnormf(4, A) == Approx{rnorm4df(A[0], A[1], A[2], A[3])});
}

TEST_CASE("(host) rnorm(double)", "[host][math][rnorm]")
{
    double A[]{0, 1, 2, 3};

    REQUIRE(rnorm(3, A) == Approx{rnorm3d(A[0], A[1], A[2])});
    REQUIRE(rnorm(4, A) == Approx{rnorm4d(A[0], A[1], A[2], A[3])});
}

TEST_CASE("(host) sincospif(float)", "[host][math][sincospif]")
{
    float s1, c1, s2, c2;
    float in1 = 1, in2 = 0.5;
    sincospif(in1, &s1, &c1);
    sincospif(in2, &s2, &c2);

    REQUIRE(s1 == Approx{0}.scale(1.0));
    REQUIRE(s2 == Approx{1});
    REQUIRE(c1 == Approx{-1});
    REQUIRE(c2 == Approx{0}.scale(1.0));
}

TEST_CASE("(host) sincospi(double)", "[host][math][sincospi]")
{
    double s1, c1, s2, c2;
    double in1 = 1, in2 = 0.5;
    sincospi(in1, &s1, &c1);
    sincospi(in2, &s2, &c2);

    CHECK(s1 == Approx{0.}.scale(1.0));
    CHECK(s2 == Approx{1.});
    CHECK(c1 == Approx{-1.});
    CHECK(c2 == Approx{0.}.scale(1.0));
}