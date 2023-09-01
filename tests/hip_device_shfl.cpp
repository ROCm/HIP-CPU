/* -----------------------------------------------------------------------------
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_fp16.h>
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

template<typename T>
__global__
void shflDownSum(T* a, int size)
{
    T val = a[threadIdx.x];
    for (int i = size / 2; i > 0; i /= 2) {
        val += __shfl_down(val, i, size);
    }
    a[threadIdx.x] = val;
}

template<typename T>
__global__
void shflUpSum(T* a, int size)
{
  T val = a[threadIdx.x];
  for (int i = size / 2; i > 0; i /= 2) {
    val += __shfl_up(val, i, size);
  }
  a[threadIdx.x] = val;
}

template<typename T>
__global__
void shflXorSum(T* a, int size)
{
  T val = a[threadIdx.x];
  for (int i = size/2; i > 0; i /= 2) {
    val += __shfl_xor(val, i, size);
  }
  a[threadIdx.x] = val;
}

inline
void getFactor(int* fact)
{
    *fact = 101;
}
inline
void getFactor(unsigned int* fact)
{
    *fact = static_cast<unsigned int>(INT32_MAX) + 1;
}
inline
void getFactor(float* fact)
{
    *fact = 2.5;
}
inline
void getFactor(double* fact)
{
    *fact = 2.5;
}
inline
void getFactor(__half* fact)
{
    *fact = 2.5;
}
inline
void getFactor(int64_t* fact)
{
    *fact = 303;
}
inline void getFactor(uint64_t* fact)
{
    *fact = static_cast<uint64_t>(INT64_MAX) + 1;
}

constexpr int sz{32};

template<typename T>
inline
T sum(T* a)
{
    T cpuSum = 0;
    T factor;
    getFactor(&factor);
    for (int i = 0; i < sz; i++) {
        a[i] = i + factor;
        cpuSum += a[i];
    }
    return cpuSum;
}

template<>
inline
__half sum(__half* a)
{
    __half cpuSum{0};
    __half factor;
    getFactor(&factor);
    for (int i = 0; i < sz; i++) {
        a[i] = i + __half2float(factor);
        cpuSum = __half2float(cpuSum) + __half2float(a[i]);
    }
    return cpuSum;
}

template<typename T>
inline
bool compare(T gpuSum, T cpuSum)
{
    if (gpuSum != cpuSum) {
        return true;
    }
    return false;
}

template<>
inline
bool compare(__half gpuSum, __half cpuSum)
{
    if (__half2float(gpuSum) != __half2float(cpuSum)) {
        return true;
    }
    return false;
}

template<typename T>
inline
void runTestShflUp()
{
    constexpr int size{32};
    T a[size];
    T cpuSum = sum(a);
    T* d_a{};

    REQUIRE(hipMalloc(&d_a, sizeof(T) * size) == hipSuccess);
    REQUIRE(
        hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault) == hipSuccess);

    hipLaunchKernelGGL(shflUpSum<T>, 1, size, 0, 0, d_a, size);

    REQUIRE(
        hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault) == hipSuccess);
    REQUIRE((compare(a[size - 1], cpuSum)) == 0);
    REQUIRE(hipFree(d_a) == hipSuccess);
}

template<typename T>
inline
void runTestShflDown()
{
    T a[sz];
    T cpuSum = sum(a);
    T* d_a;

    REQUIRE(hipMalloc(&d_a, sizeof(T) * sz) == hipSuccess);
    REQUIRE(hipMemcpy(d_a, &a, sizeof(T) * sz, hipMemcpyDefault) == hipSuccess);

    hipLaunchKernelGGL(shflDownSum<T>, 1, sz, 0, 0, d_a, sz);

    REQUIRE(hipMemcpy(&a, d_a, sizeof(T) * sz, hipMemcpyDefault) == hipSuccess);
    REQUIRE((compare(a[0], cpuSum)) == 0);
    REQUIRE(hipFree(d_a) == hipSuccess);
}

template<typename T>
inline
void runTestShflXor()
{
    T a[sz];
    T cpuSum = sum(a);
    T* d_a;

    REQUIRE(hipMalloc(&d_a, sizeof(T) * sz) == hipSuccess);
    REQUIRE(hipMemcpy(d_a, &a, sizeof(T) * sz, hipMemcpyDefault) == hipSuccess);

    hipLaunchKernelGGL(shflXorSum<T>, 1, sz, 0, 0, d_a, sz);

    REQUIRE(hipMemcpy(&a, d_a, sizeof(T) * sz, hipMemcpyDefault) == hipSuccess);
    REQUIRE((compare(a[0], cpuSum)) == 0);
    REQUIRE(hipFree(d_a) == hipSuccess);
}

TEST_CASE("Unit_runTestShfl_up", "[device][shfl]")
{
    SECTION("runTestShflUp for int") { runTestShflUp<int>(); }
    SECTION("runTestShflUp for float") { runTestShflUp<float>(); }
    SECTION("runTestShflUp for double") { runTestShflUp<double>(); }
    SECTION("runTestShflUp for __half") { runTestShflUp<__half>(); }
    SECTION("runTestShflUp for int64_t") { runTestShflUp<int64_t>(); }
    SECTION("runTestShflUp for unsigned int") { runTestShflUp<unsigned int>(); }
    SECTION("runTestShflUp for uint64_t") { runTestShflUp<uint64_t>(); }
}

TEST_CASE("Unit_runTestShfl_Down", "[device][shfl]")
{
    SECTION("runTestShflDown for int") { runTestShflDown<int>(); }
    SECTION("runTestShflDown for float") { runTestShflDown<float>(); }
    SECTION("runTestShflDown for double") { runTestShflDown<double>(); }
    SECTION("runTestShflDown for __half") { runTestShflDown<__half>(); }
    SECTION("runTestShflDown for int64_t") { runTestShflDown<int64_t>(); }
    SECTION("runTestShflDown for unsigned int") {
        runTestShflDown<unsigned int>();
    }
    SECTION("runTestShflDown for uint64_t") { runTestShflDown<uint64_t>(); }
}

TEST_CASE("Unit_runTestShfl_Xor", "[device][shfl]")
{
    SECTION("runTestShflXor for int") { runTestShflXor<int>(); }
    SECTION("runTestShflXor for float") { runTestShflXor<float>(); }
    SECTION("runTestShflXor for double") { runTestShflXor<double>(); }
    SECTION("runTestShflXor for __half") { runTestShflXor<__half>(); }
    SECTION("runTestShflXor for int64_t") { runTestShflXor<int64_t>(); }
    SECTION("runTestShflXor for unsigned int") {
        runTestShflXor<unsigned int>();
    }
    SECTION("runTestShflXor for uint64_t") { runTestShflXor<uint64_t>(); }
}