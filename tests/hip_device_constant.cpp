/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <utility>
#include <vector>

constexpr auto LEN{512u};

__constant__ int Value[LEN];

__global__
void Get(int* Ad)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Ad[tid] = Value[tid];
}

using namespace std;

TEST_CASE("hip___constant__", "[device][constant]")
{
    vector<int> A(LEN);
    generate_n(begin(A), size(A), [i = 0]() mutable { return -i++; });
    vector<int> B(LEN, 0);

    int* Ad;
    REQUIRE(hipMalloc(&Ad, sizeof(Value)) == hipSuccess);

    REQUIRE(hipMemcpyToSymbol(
        HIP_SYMBOL(Value),
        data(A),
        sizeof(Value),
        0,
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(Get, dim3(1, 1, 1), dim3(LEN, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(B), Ad, sizeof(Value), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(A == B);

    REQUIRE(hipFree(Ad) == hipSuccess);
}