/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <algorithm>
#include <vector>

constexpr auto cnt{512u};

using T = decltype(clock64());

__global__
void kernel1(T* Ad)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Ad[tid] = clock() + clock64() + __clock() + __clock64();
}

__global__
void kernel2(T* Ad)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Ad[tid] = clock() + clock64() + __clock() + __clock64() - Ad[tid];
}

using namespace std;

TEST_CASE("clock()", "[device][clock]")
{
    vector<T> A(cnt, 0);

    T* Ad;
    REQUIRE(hipMalloc((void**)&Ad, cnt * sizeof(T)) == hipSuccess);

    hipLaunchKernelGGL(kernel1, dim3(1, 1, 1), dim3(cnt, 1, 1), 0, 0, Ad);
    hipLaunchKernelGGL(kernel2, dim3(1, 1, 1), dim3(cnt, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(A), Ad, cnt * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(A), cend(A), [](auto&& x) { return x != 0; }));
}