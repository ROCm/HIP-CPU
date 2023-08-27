/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <utility>
#include <vector>

constexpr auto NUM{1024u};

__device__ int globalIn[NUM];
__device__ int globalOut[NUM];

__global__
void Assign(int* Out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Out[tid] = globalIn[tid];
    globalOut[tid] = globalIn[tid];
}

__device__ __constant__ int globalConst[NUM];

__global__
void checkAddress(int* addr, bool* out)
{
    *out = (globalConst == addr);
}

using namespace std;

TEST_CASE("hip___device__", "[device][symbol]")
{
    vector<int> A(NUM);
    generate_n(begin(A), size(A), [i = 0]() mutable { return -i++; });
    vector<int> B(NUM, 0);
    vector<int> C(NUM, 0);

    int* Am;
    REQUIRE(hipHostMalloc(&Am, sizeof(globalIn)) == hipSuccess);
    generate_n(Am, size(A), [i = 0]() mutable { return -i++; });

    int* Cm;
    REQUIRE(hipHostMalloc(&Cm, sizeof(globalIn)) == hipSuccess);
    fill_n(Cm, size(C), 0);

    int* Ad;
    REQUIRE(hipMalloc(&Ad, sizeof(globalIn)) == hipSuccess);

    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    REQUIRE(hipMemcpyToSymbolAsync(
        HIP_SYMBOL(globalIn),
        Am,
        sizeof(globalIn),
        0,
        hipMemcpyHostToDevice,
        stream) == hipSuccess);
    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);

    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(B), Ad, sizeof(globalIn), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpyFromSymbolAsync(
        Cm,
        globalOut,
        sizeof(globalOut),
        0,
        hipMemcpyDeviceToHost,
        stream) == hipSuccess);
    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);

    REQUIRE(equal(Am, Am + size(B), cbegin(B)));
    REQUIRE(equal(Am, Am + size(C), Cm));

    generate_n(begin(A), size(A), [i = 0]() mutable { return -2 * i++; });
    fill_n(begin(B), size(B), 0);

    REQUIRE(hipMemcpyToSymbol(
        HIP_SYMBOL(globalIn),
        data(A),
        sizeof(globalIn),
        0,
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(B), Ad, sizeof(globalIn), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(hipMemcpyFromSymbol(
        data(C),
        globalOut,
        sizeof(globalOut),
        0,
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(A == B);
    REQUIRE(A == C);

    generate_n(begin(A), size(A), [i = 0]() mutable { return -3 * i++; });
    fill_n(begin(B), size(B), 0);

    REQUIRE(hipMemcpyToSymbolAsync(
        HIP_SYMBOL(globalIn),
        data(A),
        sizeof(globalIn),
        0,
        hipMemcpyHostToDevice,
        stream) == hipSuccess);
    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);

    hipLaunchKernelGGL(Assign, dim3(1, 1, 1), dim3(NUM, 1, 1), 0, 0, Ad);

    REQUIRE(hipMemcpy(
        data(B), Ad, sizeof(globalIn), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpyFromSymbolAsync(
        data(C),
        globalOut,
        sizeof(globalOut),
        0,
        hipMemcpyDeviceToHost,
        stream) == hipSuccess);
    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);

    REQUIRE(A == B);
    REQUIRE(A == C);

    size_t symbolSize{};
    REQUIRE(
        hipGetSymbolSize(&symbolSize, HIP_SYMBOL(globalConst)) == hipSuccess);

    int* symbolAddress;
    REQUIRE(hipGetSymbolAddress(
        (void**)&symbolAddress, HIP_SYMBOL(globalConst)) == hipSuccess);

    bool *checkOkD;
    REQUIRE(hipMalloc(&checkOkD, sizeof(bool)) == hipSuccess);

    hipLaunchKernelGGL(
        checkAddress,
        dim3(1, 1, 1),
        dim3(1, 1, 1),
        0,
        nullptr,
        symbolAddress,
        checkOkD);

    bool checkOk{};
    REQUIRE(hipMemcpy(
        &checkOk, checkOkD, sizeof(bool), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipFree(checkOkD) == hipSuccess);
    REQUIRE(checkOk);
    REQUIRE(symbolSize == sizeof(globalConst));

    REQUIRE(hipHostFree(Am) == hipSuccess);
    REQUIRE(hipHostFree(Cm) == hipSuccess);
    REQUIRE(hipFree(Ad) == hipSuccess);
}