/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

template<typename T>
__device__
void test_integral(T* g_odata)
{

}

template<typename T>
__global__
void test_kernel(T* g_odata)
{
    const auto tid{static_cast<T>(blockDim.x * blockIdx.x + threadIdx.x)};

    atomicAdd(&g_odata[0], T{10});
    if constexpr (!is_same_v<double, T>) atomicExch(&g_odata[2], tid);

    if constexpr (is_integral_v<T>) {
        if constexpr (is_signed_v<T>) atomicSub(&g_odata[1], 10);

        atomicMax(&g_odata[3], tid);
        atomicMin(&g_odata[4], tid);
        atomicInc((unsigned int*)&g_odata[5], 17);
        atomicDec((unsigned int*)&g_odata[6], 137);
        atomicCAS(&g_odata[7], tid - 1, tid);
        atomicAnd(&g_odata[8], 2 * tid + 7);
        atomicOr(&g_odata[9], T{1} << tid);
        atomicXor(&g_odata[10], tid);
    }
}

TEMPLATE_TEST_CASE(
    "atomic*()",
    "[device][atomic]",
    int,
    unsigned int,
    unsigned long long,
    float,
    double)
{
    const unsigned int block_dim{256};
    const unsigned int grid_dim{64};
    const unsigned int lane_cnt{block_dim * grid_dim};
    const unsigned int cnt{11};

    vector<TestType> h_data(cnt, TestType{0});

    h_data[2] = TestType(cnt);
    h_data[4] = TestType(cnt);
    h_data[7] = TestType(cnt);
    h_data[8] = h_data[10] = 0xff;

    TestType* d_data;
    REQUIRE(hipMalloc((void**)&d_data, sizeof(TestType) * cnt) == hipSuccess);

    REQUIRE(hipMemcpy(
        d_data, data(h_data), sizeof(TestType) * cnt, hipMemcpyHostToDevice)
        == hipSuccess);

    hipLaunchKernelGGL(
        test_kernel, dim3(grid_dim), dim3(block_dim), 0, 0, d_data);

    REQUIRE(hipMemcpy(
        data(h_data), d_data, sizeof(TestType) * cnt, hipMemcpyDeviceToHost)
        == hipSuccess);

    SECTION("atomicAdd") { REQUIRE(h_data[0] == TestType(lane_cnt) * 10); }
    SECTION("atomicExch") {
        REQUIRE((h_data[2] >= TestType(0) && h_data[2] < TestType(lane_cnt)));
    }
    if constexpr (is_integral_v<TestType>) {
        if constexpr (is_signed_v<TestType>) {
            SECTION("atomicSub") {
                REQUIRE(h_data[1] == TestType(lane_cnt) * -10);
            }
        }
        SECTION("atomicMax") { REQUIRE(h_data[3] == lane_cnt - 1); }
        SECTION("atomicMin") { REQUIRE(h_data[4] == 0); }
        SECTION("atomicInc") { REQUIRE_FALSE(h_data[5] == cnt % 17); }
        SECTION("atomicDec") { REQUIRE_FALSE(h_data[6] == 0); }
        SECTION("atomicCAS") {
            REQUIRE(
                (h_data[7] >= TestType(0) && h_data[7] < TestType(lane_cnt)));
        }
        SECTION("atomicAnd") { REQUIRE(h_data[8] == 1); }
        SECTION("atomicOr") {  REQUIRE(h_data[9] == TestType(UINT64_MAX)); }
        SECTION("atomicXor") { REQUIRE(h_data[10] == 0xff); }
    }

    hipFree(d_data);
}