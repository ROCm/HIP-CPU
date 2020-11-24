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
void testExternSharedKernel(
    const T* A_d,
    const T* B_d,
    T* C_d,
    size_t numElements,
    size_t groupElements)
{
    HIP_DYNAMIC_SHARED(T, sdata)

    const auto tid{threadIdx.x};

    if (tid < groupElements) sdata[tid] = static_cast<T>(tid);

    __syncthreads();

    unsigned int pout{0u};
    unsigned int pin{1u};

    for (auto dx = 1u; dx < groupElements; dx *= 2u) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid < groupElements) {
            if (tid >= dx) {
                sdata[pout * groupElements + tid] =
                    sdata[pin * groupElements + tid] +
                    sdata[pin * groupElements + tid - dx];
            }
            else {
                sdata[pout * groupElements + tid] =
                    sdata[pin * groupElements + tid];
            }
        }

        __syncthreads();
    }

    const auto gid{(blockIdx.x * blockDim.x + threadIdx.x)};
    C_d[gid] = A_d[gid] + B_d[gid] + sdata[pout * groupElements + (tid % groupElements)];
}

constexpr auto threads_per_block{256u};

TEMPLATE_TEST_CASE(
    "Dynamic SharedMem I", "[device][dynamic_shared_1]", float, double)
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    const auto N{GENERATE(1024, 65536)};
    const auto groupElements{GENERATE(4, 8, 16, 32, 64)};

    vector<TestType> A(N);
    for (auto i = 0u; i != size(A); ++i) A[i] = TestType{3.146} * i;
    vector<TestType> B(N);
    for (auto i = 0u; i != size(B); ++i) B[i] = TestType{1.618} * i;
    vector<TestType> C(N);
    for (auto i = 0u; i != size(C); ++i) C[i] = static_cast<TestType>(i);

    TestType* A_d;
    TestType* B_d;
    TestType* C_d;

    REQUIRE(hipMalloc((void**)&A_d, size(A) * sizeof(TestType)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&B_d, size(B) * sizeof(TestType)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&C_d, size(C) * sizeof(TestType)) == hipSuccess);

    REQUIRE(hipMemcpy(
        A_d,
        data(A),
        size(A) * sizeof(TestType),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        B_d,
        data(B),
        size(B) * sizeof(TestType),
        hipMemcpyHostToDevice) == hipSuccess);

    const dim3 blocks{(N + threads_per_block - 1u) / threads_per_block};
    const auto groupMemBytes{2 * groupElements * sizeof(TestType)};

    // launch kernel with dynamic shared memory
    hipLaunchKernelGGL(
        testExternSharedKernel,
        dim3(blocks),
        dim3(threads_per_block),
        groupMemBytes,
        0,
        A_d,
        B_d,
        C_d,
        N,
        groupElements);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hipMemcpy(
        data(C),
        C_d,
        size(C) * sizeof(TestType),
        hipMemcpyDeviceToHost) == hipSuccess);

    for (decltype(size(C)) i = 0u; i != size(C); ++i) {
        const auto tid{i % groupElements};
        const auto sumFromSMEM{static_cast<TestType>(tid * (tid + 1) / 2)};
        INFO("i = " << i);
        REQUIRE(C[i] == Approx{A[i] + B[i] + sumFromSMEM});
    }

    REQUIRE(hipFree(A_d) == hipSuccess);
    REQUIRE(hipFree(B_d) == hipSuccess);
    REQUIRE(hipFree(C_d) == hipSuccess);
}