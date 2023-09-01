/* -----------------------------------------------------------------------------
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <cstdlib>

using namespace std;

__global__
void warpvote(int* device_any, int* device_all, int pshift)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    device_any[threadIdx.x >> pshift] = __any(tid - 77);
    device_all[threadIdx.x >> pshift] = __all(tid - 77);
}

TEST_CASE("Unit_AnyAll_CompileTest", "[device][all][any]")
{
    auto w{warpSize};
    auto pshift{0};

    while (w >>= 1) ++pshift;

    INFO("WarpSize: " << warpSize << " pShift: " << pshift);

    auto anycount{0};
    auto allcount{0};
    auto Num_Threads_per_Block{1024};
    auto Num_Blocks_per_Grid{1};
    auto Num_Warps_per_Grid{
        (Num_Threads_per_Block * Num_Blocks_per_Grid) / warpSize};

    auto host_any{static_cast<int*>(malloc(Num_Warps_per_Grid * sizeof(int)))};
    auto host_all{static_cast<int*>(malloc(Num_Warps_per_Grid * sizeof(int)))};

    int* device_any{};
    int* device_all{};

    REQUIRE(
        hipMalloc(&device_any, Num_Warps_per_Grid * sizeof(int)) == hipSuccess);
    REQUIRE(
        hipMalloc(&device_all, Num_Warps_per_Grid * sizeof(int)) == hipSuccess);

    fill_n(host_any, Num_Warps_per_Grid, 0);
    fill_n(host_all, Num_Warps_per_Grid, 0);

    REQUIRE(hipMemcpy(
        device_any,
        host_any,
        sizeof(int),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        device_all,
        host_all,
        sizeof(int),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        warpvote,
        dim3(Num_Blocks_per_Grid),
        dim3(Num_Threads_per_Block),
        0,
        nullptr,
        device_any,
        device_all,
        pshift);

    REQUIRE(hipGetLastError() == hipSuccess);
    REQUIRE(hipMemcpy(
        host_any,
        device_any,
        Num_Warps_per_Grid * sizeof(int),
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        host_all,
        device_all,
        Num_Warps_per_Grid * sizeof(int),
        hipMemcpyDeviceToHost) == hipSuccess);

    for (int i = 0; i < Num_Warps_per_Grid; i++) {
        INFO(
            "Warp Number: " << i << " __any: " << host_any[i]
                << " __all: " << host_all[i]);

        if (host_all[i] != 1) ++allcount;
        if (host_any[i] != 1) {
            ++anycount;
        }
    }

    REQUIRE(hipFree(device_any) == hipSuccess);
    REQUIRE(hipFree(device_all) == hipSuccess);

    REQUIRE(anycount == 0);
    REQUIRE(allcount == 1);
}