/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdlib>
#include <vector>

using namespace std;

template<typename T>
__global__
void testDeviceBarrierNotInFirstBlock(
    T* Out_d,
    size_t numElements,
    size_t groupElements)
{
    HIP_DYNAMIC_SHARED(T, sdata)

    const auto tid{threadIdx.x};

    if(blockIdx.x == 0){
        Out_d[tid] = static_cast<T>(tid);
        return;
    }

    if (tid < groupElements) sdata[tid] = static_cast<T>(tid);

    __syncthreads();

    const auto gid{(blockIdx.x * blockDim.x + threadIdx.x)};
    Out_d[gid] = sdata[(tid + 1) % groupElements];
}

constexpr auto threads_per_block{256u};

TEMPLATE_TEST_CASE(
    "Barrier not in first block", "[device][barrier_not_in_first_block]", float, double)
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    const auto N{GENERATE(1024, 65536)};
    const auto groupElements{GENERATE(4, 8, 16, 32, 64)};

    vector<TestType> Out(N);

    TestType* Out_d;

    REQUIRE(hipMalloc(&Out_d, size(Out) * sizeof(TestType)) == hipSuccess);

    const dim3 blocks{(N + threads_per_block - 1u) / threads_per_block};
    const auto groupMemBytes{2 * groupElements * sizeof(TestType)};

    // launch kernel with dynamic shared memory
    hipLaunchKernelGGL(
        testDeviceBarrierNotInFirstBlock,
        dim3(blocks),
        dim3(threads_per_block),
        static_cast<std::uint32_t>(groupMemBytes),
        0,
        Out_d,
        N,
        groupElements);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hipMemcpy(
        data(Out),
        Out_d,
        size(Out) * sizeof(TestType),
        hipMemcpyDeviceToHost) == hipSuccess);

    for (decltype(size(Out)) i = 0u; i != size(Out); ++i) {
        const auto tid{i % groupElements};
        INFO("i = " << i);
        REQUIRE(Out[i] == Approx{i < threads_per_block ? i : (tid + 1) % groupElements});
    }

    REQUIRE(hipFree(Out_d) == hipSuccess);
}
