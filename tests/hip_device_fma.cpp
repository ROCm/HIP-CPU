/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <vector>

using namespace std;

__global__
void kernelTestFMA(bool* Ad)
{
    const float f = 1.0f / 3.0f;
    const double d = f;

    // f * f + 3.0f will be different if promoted to double.
    const float floatResult = fma(f, f, 3.0f);
    const double doubleResult = fma(d, d, 3.0);

    if (floatResult == doubleResult) return;

    // check promote to float.
    unsigned int n{0u};

    Ad[n++] = fma(f, f, 3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (char)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (unsigned char)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (short)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (unsigned short)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (int)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (unsigned int)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (long)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, (unsigned long)3) == Approx{floatResult};
    Ad[n++] = fma(f, f, true) == Approx{fma(f, f, 1.0f)};

    // check promote to double.
    Ad[n++] = fma(d, d, 3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (char)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (unsigned char)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (short)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (unsigned short)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (int)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (unsigned int)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (long)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, (unsigned long)3) == Approx{floatResult};
    Ad[n++] = fma(d, d, true) == Approx{fma(d, d, 1.0f)};
}

constexpr auto cnt_flt_dbl{20};

TEST_CASE("fma(float)", "[device][math][fma]")
{
    array<bool, cnt_flt_dbl> A{};

    bool* Ad;

    REQUIRE(hipMalloc((void **)&Ad, sizeof(bool) * size(A)) == hipSuccess);

    hipLaunchKernelGGL(kernelTestFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(A),
        Ad,
        sizeof(bool) * size(A),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(A), cend(A), [](auto&& x) { return x; }));
}

__global__
void kernelTestHalfFMA(bool* Ad)
{
    __half h = (__half)(1.0f/3.0f);
    float f = h;
    double d = f;

    // h * h + 3 will be different if promoted to float.
    __half halfResult = fma(h, h, (__half)3);
    float floatResult = fma(f, f, 3.0f);
    double doubleResult = fma(d, d, 3.0);

    if (halfResult == Approx{floatResult}) return;
    if (halfResult == Approx{doubleResult}) return;

    // check promote to half.
    unsigned int n{0u};

    Ad[n++] = fma(h, h, 3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (char)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (unsigned char)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (short)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (unsigned short)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (int)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (unsigned int)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (long)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, (unsigned long)3) == Approx{floatResult};
    Ad[n++] = fma(h, h, true) == Approx{fma(f, f, 1.0f)};
}

constexpr auto cnt_hlf{10};

TEST_CASE("fma(__half)", "[device][math][fma]")
{
    array<bool, cnt_hlf> A{};

    bool* Ad;

    REQUIRE(hipMalloc((void **)&Ad, sizeof(bool) * size(A)) == hipSuccess);

    hipLaunchKernelGGL(
        kernelTestHalfFMA, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(A),
        Ad,
        sizeof(bool) * size(A),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(A), cend(A), [](auto&& x) { return x; }));
}