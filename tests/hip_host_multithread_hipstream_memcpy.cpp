/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdlib>
#include <vector>
#include <utility>

using namespace std;

constexpr auto N{1000};

template<typename T>
__global__
void Inc(T* Array)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    Array[tx] = Array[tx] + T(1);
}

void run1(hipStream_t stream)
{
    vector<float> A(N, 1.0f);
    vector<float> B(N);
    vector<float> E(N);

    float* Cd;
    float* Dd;

    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpyAsync(
        data(B),
        data(A),
        sizeof(float) * size(A),
        hipMemcpyHostToHost,
        stream) == hipSuccess);
    REQUIRE(hipMemcpyAsync(
        Cd,
        data(B),
        sizeof(float) * size(A),
        hipMemcpyHostToDevice,
        stream) == hipSuccess);

    hipLaunchKernelGGL(Inc, dim3(N / 500), dim3(500), 0, stream, Cd);

    REQUIRE(hipMemcpyAsync(
        Dd, Cd, sizeof(float) * N, hipMemcpyDeviceToDevice, stream) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        data(E), Dd, sizeof(float) * size(E), hipMemcpyDeviceToHost, stream) ==
        hipSuccess);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    for (decltype(size(A)) i = 0; i != size(A); ++i) {
        REQUIRE(Approx{E[i]} == A[i] + 1.f);
    }
}


void run(hipStream_t stream1, hipStream_t stream2)
{
    vector<float> A(N, 42.f);
    vector<float> AA(N, 69.f);
    vector<float> B(N);
    vector<float> BB(N);
    vector<float> E(N);
    vector<float> EE(N);

    float* Cd;
    float* Cdd;
    float* Dd;
    float* Ddd;

    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cdd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Ddd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpyAsync(
        data(B), data(A), sizeof(float) * N, hipMemcpyHostToHost, stream1) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        data(BB), data(AA), sizeof(float) * N, hipMemcpyHostToHost, stream2) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        Cd, data(B), sizeof(float) * N, hipMemcpyHostToDevice, stream1) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        Cdd, data(BB), sizeof(float) * N, hipMemcpyHostToDevice, stream2) ==
        hipSuccess);

    hipLaunchKernelGGL(Inc, dim3(N / 500), dim3(500), 0, stream1, Cd);
    hipLaunchKernelGGL(Inc, dim3(N / 500), dim3(500), 0, stream2, Cdd);

    REQUIRE(hipMemcpyAsync(
        Dd, Cd, sizeof(float) * N, hipMemcpyDeviceToDevice, stream1) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        Ddd, Cdd, sizeof(float) * N, hipMemcpyDeviceToDevice, stream2) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        data(E), Dd, sizeof(float) * N, hipMemcpyDeviceToHost, stream1) ==
        hipSuccess);
    REQUIRE(hipMemcpyAsync(
        data(EE), Ddd, sizeof(float) * N, hipMemcpyDeviceToHost, stream2) ==
        hipSuccess);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    for (decltype(size(E)) i = 0u; i != size(E); ++i) {
        REQUIRE(Approx{E[i]} == A[i] + 1.f);
        REQUIRE(Approx{EE[i]} == AA[i] + 1.f);
    }

    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Cdd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
    REQUIRE(hipFree(Ddd) == hipSuccess);
}

TEST_CASE(
    "Parallel, async memcpy, multiple streams, many operations.",
    "[host][multithread][hipStream_t]")
{
    constexpr auto iterations{100};

    hipStream_t stream[3]{};
    for (auto&& x : stream) REQUIRE(hipStreamCreate(&x) == hipSuccess);

    const size_t size = N * sizeof(float);

    for (auto i = 0; i != iterations; ++i) {
        std::thread t1{run1, stream[0]};
        std::thread t2{run1, stream[0]};
        std::thread t3{run, stream[1], stream[2]};

        t1.join();
        t2.join();
        t3.join();
    }

    for (auto&& x : stream) REQUIRE(hipStreamDestroy(x) == hipSuccess);
}