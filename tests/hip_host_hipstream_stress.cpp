/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <climits>
#include <vector>

#define _SIZE sizeof(int) * 1024 * 1024
#define NUM_STREAMS 20
#define ITER 1 << 10

__global__
void Iter(int* Ad, int num)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx == 0) {
        for (int i = 0; i < num; i++) {
            Ad[tx] += 1;
        }
    }
}

using namespace std;

TEST_CASE("Many streams stress test.", "[host][hipStream_t]")
{
    constexpr auto cnt{1024 * 1024};
    constexpr auto n{1 << 10};
    constexpr auto stream_cnt{20};

    vector<int*> A(stream_cnt);
    for (auto&& x : A) {
        REQUIRE(hipHostMalloc((void**)&x, sizeof(int) * cnt) == hipSuccess);
        x[0] = 1;
    }
    vector<int*> Ad(stream_cnt);
    for (auto&& x : Ad) {
        REQUIRE(hipMalloc((void**)&x, sizeof(int) * cnt) == hipSuccess);
    }

    vector<hipStream_t> stream(stream_cnt);
    for (auto&& x : stream) REQUIRE(hipStreamCreate(&x) == hipSuccess);

    for (auto i = 0; i != stream_cnt; ++i) {
        for (auto j = 0; j != n; ++j) {
            REQUIRE(hipMemcpyAsync(
                Ad[i],
                A[i],
                sizeof(int) * cnt,
                hipMemcpyHostToDevice,
                stream[i]) == hipSuccess);

            hipLaunchKernelGGL(
                Iter, dim3(1), dim3(1), 0, stream[i], Ad[i], SHRT_MAX);

            REQUIRE(hipMemcpyAsync(
                A[i],
                Ad[i],
                sizeof(int) * cnt,
                hipMemcpyDeviceToHost,
                stream[i]) == hipSuccess);
        }
    }

    REQUIRE(hipDeviceSynchronize() == hipSuccess);
}