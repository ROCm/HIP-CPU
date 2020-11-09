/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstring>
#include <limits>
#include <vector>

__global__
void __halfMath(bool* result, __half a)
{
  result[0] = __heq(__hadd(a, __half{1}), __half{2});
  result[0] = __heq(__hadd_sat(a, __half{1}), __half{1}) && result[0];
  result[0] = __heq(__hfma(a, __half{2}, __half{3}), __half{5}) && result[0];
  result[0] =
    __heq(__hfma_sat(a, __half{2}, __half{3}), __half{1}) && result[0];
  result[0] = __heq(__hsub(a, __half{1}), __half{0}) && result[0];
  result[0] = __heq(__hsub_sat(a, __half{2}), __half{0}) && result[0];
  result[0] = __heq(__hmul(a, __half{2}), __half{2}) && result[0];
  result[0] = __heq(__hmul_sat(a, __half{2}), __half{1}) && result[0];
  result[0] = __heq(__hdiv(a, __half{2}), __half{0.5}) && result[0];
}

__device__
bool to_bool(const __half2& x)
{
  return x != 0;
}

__global__
void __half2Math(bool* result, __half2 a)
{
  result[0] =
    to_bool(__heq2(__hadd2(a, __half2{1, 1}), __half2{2, 2}));
  result[0] = to_bool(__heq2(__hadd2_sat(a, __half2{1, 1}), __half2{1, 1})) &&
    result[0];
  result[0] = to_bool(__heq2(
    __hfma2(a, __half2{2, 2}, __half2{3, 3}), __half2{5, 5})) && result[0];
  result[0] = to_bool(__heq2(
    __hfma2_sat(a, __half2{2, 2}, __half2{3, 3}), __half2{1, 1})) && result[0];
  result[0] = to_bool(__heq2(__hsub2(a, __half2{1, 1}), __half2{0, 0})) &&
    result[0];
  result[0] = to_bool(__heq2(__hsub2_sat(a, __half2{2, 2}), __half2{0, 0})) &&
    result[0];
  result[0] = to_bool(__heq2(__hmul2(a, __half2{2, 2}), __half2{2, 2})) &&
    result[0];
  result[0] = to_bool(__heq2(__hmul2_sat(a, __half2{2, 2}), __half2{1, 1})) &&
    result[0];
  result[0] = to_bool(__heq2(__h2div(a, __half2{2, 2}), __half2{0.5, 0.5})) &&
    result[0];
}

__global__
void kernel_hisinf(__half* input, bool* output)
{
  int tx = threadIdx.x;
  output[tx] = __hisinf(input[tx]);
}

__global__
void kernel_hisnan(__half* input, bool* output)
{
  int tx = threadIdx.x;
  output[tx] = __hisnan(input[tx]);
}

__global__
void testHalfAbs(float* p)
{
    auto a = __float2half(*p);
    a = __habs(a);
    *p = __half2float(a);
}

__global__
void testHalf2Abs(float2* p)
{
    auto a = __float22half2_rn(*p);
    a = __habs2(a);
    *p = __half22float2(a);
}

__half host_ushort_as_half(unsigned short s)
{
    __half r{};
    std::memcpy(&r, &s, sizeof(r));

    return r;
}

TEST_CASE("isinf(__half)", "[device][math][half]")
{
    using namespace std;

    const vector<__half> in{
        numeric_limits<__half>::infinity(),
        -numeric_limits<__half>::infinity(),
        numeric_limits<__half>::signaling_NaN(),
        numeric_limits<__half>::quiet_NaN(),
        -numeric_limits<__half>::signaling_NaN(),
        -numeric_limits<__half>::quiet_NaN(),
        host_ushort_as_half(0x0000), // 0
        host_ushort_as_half(0x8000), // -0
        host_ushort_as_half(0x7bff), // max +ve normal
        host_ushort_as_half(0xfbff), // max -ve normal
        host_ushort_as_half(0x0400), // min +ve normal
        host_ushort_as_half(0x8400), // min -ve normal
        host_ushort_as_half(0x03ff), // max +ve sub-normal
        host_ushort_as_half(0x83ff), // max -ve sub-normal
        host_ushort_as_half(0x0001), // min +ve sub-normal
        host_ushort_as_half(0x8001)  // min -ve sub-normal
    };

    __half* d_in{};
    REQUIRE(hipMalloc((void**)&d_in, sizeof(__half) * size(in)) == hipSuccess);
    REQUIRE(hipMemcpy(
        d_in,
        data(in),
        sizeof(__half) * size(in),
        hipMemcpyHostToDevice) == hipSuccess);

    bool* d_out{};
    REQUIRE(hipMalloc((void**)&d_out, sizeof(bool) * size(in)) == hipSuccess);

    hipLaunchKernelGGL(
        kernel_hisinf, dim3(1), dim3(size(in)), 0, 0, d_in, d_out);
    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(d_out[0]);
    REQUIRE(d_out[1]);
    REQUIRE(none_of(d_out + 2, d_out + size(in), [](auto&& x) { return x; }));

    // free memory
    REQUIRE(hipFree(d_out) == hipSuccess);
    REQUIRE(hipFree(d_in) == hipSuccess);
}

TEST_CASE("isnan(__half)", "[device][math][half]")
{
    using namespace std;

    const vector<__half> in{
        numeric_limits<__half>::infinity(),
        -numeric_limits<__half>::infinity(),
        numeric_limits<__half>::signaling_NaN(),
        numeric_limits<__half>::quiet_NaN(),
        -numeric_limits<__half>::signaling_NaN(),
        -numeric_limits<__half>::quiet_NaN(),
        host_ushort_as_half(0x0000), // 0
        host_ushort_as_half(0x8000), // -0
        host_ushort_as_half(0x7bff), // max +ve normal
        host_ushort_as_half(0xfbff), // max -ve normal
        host_ushort_as_half(0x0400), // min +ve normal
        host_ushort_as_half(0x8400), // min -ve normal
        host_ushort_as_half(0x03ff), // max +ve sub-normal
        host_ushort_as_half(0x83ff), // max -ve sub-normal
        host_ushort_as_half(0x0001), // min +ve sub-normal
        host_ushort_as_half(0x8001)  // min -ve sub-normal
    };

    __half* d_in{};
    REQUIRE(hipMalloc((void**)&d_in, sizeof(__half) * size(in)) == hipSuccess);
    REQUIRE(hipMemcpy(
        d_in,
        data(in),
        sizeof(__half) * size(in),
        hipMemcpyHostToDevice) == hipSuccess);

    bool* d_out{};
    REQUIRE(hipMalloc((void**)&d_out, sizeof(bool) * size(in)) == hipSuccess);

    hipLaunchKernelGGL(
        kernel_hisnan, dim3(1), dim3(size(in)), 0, 0, d_in, d_out);
    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(d_out[2]);
    REQUIRE(d_out[3]);
    REQUIRE(d_out[4]);
    REQUIRE(d_out[5]);
    REQUIRE(none_of(d_out, d_out + 2, [](auto&& x) { return x; }));
    REQUIRE(none_of(d_out + 6, d_out + size(in), [](auto&& x) { return x; }));

    // free memory
    REQUIRE(hipFree(d_out) == hipSuccess);
    REQUIRE(hipFree(d_in) == hipSuccess);
}

TEST_CASE("abs(__half)", "[device][math][half]")
{
    float* p;
    REQUIRE(hipMalloc((void**)&p, sizeof(float)) == hipSuccess);

    float pp{-2.1f};
    REQUIRE(hipMemcpy(p, &pp, sizeof(float), hipMemcpyDefault) == hipSuccess);

    hipLaunchKernelGGL(testHalfAbs, 1, 1, 0, 0, p);

    REQUIRE(hipMemcpy(&pp, p, sizeof(float), hipMemcpyDefault) == hipSuccess);
    REQUIRE(hipFree(p) == hipSuccess);

    REQUIRE(pp >= 0.0f);
}

TEST_CASE("abs(__half2)", "[device][math][half]")
{
    float2 *p;
    REQUIRE(hipMalloc((void**)&p, sizeof(float2)) == hipSuccess);

    float2 pp{-2.1f, -1.1f};
    REQUIRE(hipMemcpy(p, &pp, sizeof(float2), hipMemcpyDefault) == hipSuccess);

    hipLaunchKernelGGL(testHalf2Abs, 1, 1, 0, 0, p);

    REQUIRE(hipMemcpy(&pp, p, sizeof(float2), hipMemcpyDefault) == hipSuccess);
    REQUIRE(hipFree(p) == hipSuccess);

    REQUIRE(pp.x >= 0.0f);
    REQUIRE(pp.y >= 0.0f);
}

TEST_CASE("math_func(__half)", "[device][math][half]")
{
    bool* result{nullptr};
    REQUIRE(hipHostMalloc((void**)&result, sizeof(result)) == hipSuccess);

    result[0] = false;

    hipLaunchKernelGGL(__halfMath, dim3(1), dim3(1), 0, 0, result, __half{1});
    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(result[0]);

    REQUIRE(hipHostFree(result) == hipSuccess);
}

TEST_CASE("math_func(__half2)", "[device][math][half]")
{
    bool* result{nullptr};
    REQUIRE(hipHostMalloc((void**)&result, sizeof(result)) == hipSuccess);

    result[0] = false;


    hipLaunchKernelGGL(
        __half2Math,dim3(1), dim3(1), 0, 0, result, __half2{1, 1});
    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(result[0]);

    REQUIRE(hipHostFree(result) == hipSuccess);
}