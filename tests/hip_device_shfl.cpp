/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

constexpr auto WIDTH{4u};
constexpr auto NUM{WIDTH * WIDTH};

constexpr auto THREADS_PER_BLOCK_X{4u};
constexpr auto THREADS_PER_BLOCK_Y{4u};
constexpr auto THREADS_PER_BLOCK_Z{1u};

namespace
{
    template<typename T>
    inline
    constexpr
    T getFactor() noexcept
    {
        if constexpr (is_same_v<T, float>) return 2.5;
        else if constexpr (is_same_v<T, double>) return 2.5;
        else if constexpr (is_same_v<T, int>) return 101;
        else if constexpr (is_same_v<T, unsigned int>) return INT32_MAX + 1u;
        else if constexpr (is_same_v<T, long>) return 202;
        else if constexpr (is_same_v<T, unsigned long>) return LONG_MAX + 1ul;
        else if constexpr (is_same_v<T, long long>) return 303;
        else return LLONG_MAX + 1ull;
    }

    template<typename T>
    __global__
    void matrixTranspose(T* out, T* in, unsigned int width)
    {
        const auto x{blockDim.x * blockIdx.x + threadIdx.x};
        auto val = in[x];
        for (auto i = 0u; i != width; ++i) {
            for (auto j = 0u; j != width; ++j) {
                out[i * width + j] = __shfl(val, j * width + i);
            }
        }
    }

    template<typename T>
    inline
    void matrixTransposeCPUReference(T* output, T* input, unsigned int width)
    {
        for (auto j = 0u; j != width; ++j) {
            for (auto i = 0u; i != width; ++i) {
                output[i * width + j] = input[j * width + i];
            }
        }
    }
} // Unnamed namespace

TEMPLATE_TEST_CASE(
    "shfl()",
    "[device][shfl]",
    int,
    float,
    double,
    long,
    long long,
    unsigned int,
    unsigned long,
    unsigned long long)
{
    vector<TestType> Matrix(NUM);
    generate_n(begin(Matrix), size(Matrix), [i = TestType{0}]() mutable {
        return getFactor<TestType>() + i++;
    });

    TestType* GPUMatrix;
    TestType* GPUTransposeMatrix;

    REQUIRE(hipMalloc(&GPUMatrix, NUM * sizeof(TestType)) == hipSuccess);
    REQUIRE(hipMalloc(
        &GPUTransposeMatrix, NUM * sizeof(TestType)) == hipSuccess);

    REQUIRE(hipMemcpy(
        GPUMatrix,
        data(Matrix),
        NUM * sizeof(TestType),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        matrixTranspose<TestType>,
        dim3(1),
        dim3(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y),
        0,
        nullptr,
        GPUTransposeMatrix,
        GPUMatrix,
        WIDTH);

    vector<TestType> TransposeMatrix(NUM);
    REQUIRE(hipMemcpy(
        data(TransposeMatrix),
        GPUTransposeMatrix,
        NUM * sizeof(TestType),
        hipMemcpyDeviceToHost) == hipSuccess);

    vector<TestType> CPUTransposeMatrix(NUM);
    matrixTransposeCPUReference(data(CPUTransposeMatrix), data(Matrix), WIDTH);

    if constexpr (is_floating_point_v<TestType>) {
        REQUIRE(equal(
            cbegin(TransposeMatrix),
            cend(TransposeMatrix),
            cbegin(CPUTransposeMatrix),
            [](auto&& x, auto&& y) { return Approx{x} == y; }));
    }
    else REQUIRE(TransposeMatrix == CPUTransposeMatrix);

    REQUIRE(hipFree(GPUMatrix) == hipSuccess);
    REQUIRE(hipFree(GPUTransposeMatrix) == hipSuccess);

    // TODO: add tests for OOB indices.
}