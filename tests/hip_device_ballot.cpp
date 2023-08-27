/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

__global__
void gpu_ballot(
    unsigned int* device_ballot, int Num_Warps_per_Block, int pshift)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int warp_num = threadIdx.x >> pshift;

    if constexpr (warpSize > sizeof(std::uint32_t) * CHAR_BIT) {
        atomicAdd(
            &device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popcll(__ballot(tid - 245)));
    }
    else {
        atomicAdd(
            &device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popc(static_cast<std::uint32_t>(__ballot(tid - 245))));
    }
}

using namespace std;

TEST_CASE("ballot()", "[device][ballot]")
{
    int w = warpSize;
    int pshift = 0;
    while (w >>= 1) ++pshift;

    unsigned int Num_Threads_per_Block = 512;
    unsigned int Num_Blocks_per_Grid = 1;
    unsigned int Num_Warps_per_Block = Num_Threads_per_Block / warpSize;
    unsigned int Num_Warps_per_Grid =
        (Num_Threads_per_Block * Num_Blocks_per_Grid) / warpSize;

    vector<unsigned int> host_ballot(Num_Warps_per_Grid, 0u);

    unsigned int* device_ballot;
    REQUIRE(hipMalloc(
        &device_ballot, Num_Warps_per_Grid * sizeof(unsigned int)) ==
        hipSuccess);

    REQUIRE(hipMemcpy(
        device_ballot,
        data(host_ballot),
        Num_Warps_per_Grid * sizeof(unsigned int),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        gpu_ballot,
        dim3(Num_Blocks_per_Grid),
        dim3(Num_Threads_per_Block),
        0,
        0,
        device_ballot,
        Num_Warps_per_Block,
        pshift);


    REQUIRE(hipMemcpy(
        data(host_ballot),
        device_ballot,
        Num_Warps_per_Grid * sizeof(unsigned int),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(count_if(cbegin(host_ballot), cend(host_ballot), [=](auto&& x) {
        return (x != 0) && (x / warpSize != warpSize);
    }) == 1);

    REQUIRE(hipFree(device_ballot) == hipSuccess);
}