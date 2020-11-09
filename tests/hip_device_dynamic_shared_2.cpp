/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <utility>
#include <vector>

#define LEN 16 * 1024
#define SIZE LEN * 4

__global__
void vectorAdd(float* Ad, float* Bd)
{
    HIP_DYNAMIC_SHARED(float, sBd);
    int tx = threadIdx.x;
    for (int i = 0; i < LEN / 64; i++) {
        sBd[tx + i * 64] = Ad[tx + i * 64] + 1.0f;
        Bd[tx + i * 64] = sBd[tx + i * 64];
    }
}

using namespace std;

TEST_CASE("Dynamic SharedMem II", "[device][dynamic_shared]")
{
    vector<float> A(LEN, 1.f);
    vector<float> B(LEN, 0.f);

    float* Ad;
    float* Bd;

    REQUIRE(hipMalloc((void**)&Ad, size(A) * sizeof(float)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, size(B) * sizeof(float)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        size(A) * sizeof(float),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        size(B) * sizeof(float),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        vectorAdd, dim3(1), dim3(64), size(A) * sizeof(float), 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        size(B) * sizeof(float),
        hipMemcpyDeviceToHost) == hipSuccess);

    for (decltype(size(A)) i = 0; i != size(A); ++i) {
        REQUIRE(B[i] == Approx{A[i] + 1.f});
    }
}