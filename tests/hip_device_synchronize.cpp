/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

constexpr auto byte_cnt{sizeof(int) * 1024 * 1024};
constexpr auto stream_cnt{2};

__global__
void Iter(int* Ad, int num)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tx) return;

    // Kernel loop designed to execute very slowly... ... ...   so we can test
    // timing-related behaviour below
    for (int i = 0; i != num; ++i) ++Ad[tx];
}

TEST_CASE("hipDeviceSynchronize()", "[host][hipDevice]")
{
    int* A[stream_cnt];
    int* Ad[stream_cnt];
    hipStream_t stream[stream_cnt];

    for (auto i = 0; i != stream_cnt; ++i) {
        REQUIRE(hipHostMalloc(
            (void**)&A[i], byte_cnt, hipHostMallocDefault) == hipSuccess);

        A[i][0] = 1;

        REQUIRE(hipMalloc((void**)&Ad[i], byte_cnt) == hipSuccess);
        REQUIRE(hipStreamCreate(&stream[i]) == hipSuccess);
    }

    for (auto i = 0; i != stream_cnt; ++i) {
        REQUIRE(hipMemcpyAsync(
            Ad[i], A[i], byte_cnt, hipMemcpyHostToDevice, stream[i]) ==
            hipSuccess);
    }
    for (auto i = 0; i != stream_cnt; ++i) {
        hipLaunchKernelGGL(
            Iter, dim3(1), dim3(1), 0, stream[i], Ad[i], 1 << 30);
    }
    for (auto i = 0; i != stream_cnt; ++i) {
        REQUIRE(hipMemcpyAsync(
            A[i], Ad[i], byte_cnt, hipMemcpyDeviceToHost, stream[i]) ==
            hipSuccess);
    }

    // This first check but relies on the kernel running for so long that the
    // D2H async memcopy has not started yet. This will be true in an optimal
    // asynchronous implementation. Conservative implementations which
    // synchronize the hipMemcpyAsync will fail.
    REQUIRE(A[stream_cnt - 1][0] - 1 != 1 << 30);
    REQUIRE(hipDeviceSynchronize() == hipSuccess);
    REQUIRE(A[stream_cnt - 1][0] - 1 == 1 << 30);
}