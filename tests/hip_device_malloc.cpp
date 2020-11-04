/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <functional>
#include <utility>
#include <vector>

constexpr auto block_dim_x{64};
constexpr auto block_dim_y{1};
constexpr auto block_dim_z{1};
constexpr auto grid_dim_x{1};
constexpr auto grid_dim_y{1};
constexpr auto grid_dim_z{1};

constexpr auto cnt{
    grid_dim_x * block_dim_x *
    grid_dim_y * block_dim_y *
    grid_dim_z * block_dim_z};

// Allocate memory in kernel and save the address to pA and pB.
// Copy value from A, B to allocated memory.
template<typename T>
__global__
void kernel_alloc(T* A, T* B, T** pA, T** pB)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x +
        (threadIdx.y + blockDim.y * blockIdx.y) * blockDim.x +
        (threadIdx.z + blockDim.z * blockIdx.z) * blockDim.x * blockDim.y;

    if (tx) return;

    *pA = (T*)malloc(sizeof(T) * cnt);
    *pB = (T*)malloc(sizeof(T) * cnt);
    for (auto i = 0; i != cnt; ++i) {
        (*pA)[i] = A[i];
        (*pB)[i] = B[i];
    }
}

// Do calculation using values saved in allocated memory. pA, pB are buffers
// containing the address of the device-side allocated array.
template<typename T, typename F>
__global__
void kernel_free(T** pA, T** pB, T* C, F fn)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x +
        (threadIdx.y + blockDim.y * blockIdx.y) * blockDim.x +
        (threadIdx.z + blockDim.z * blockIdx.z) * blockDim.x * blockDim.y;
    C[tx] = fn((*pA)[tx], (*pB)[tx]);

    if (tx != blockDim.x - 1) return;

    free(*pA);
    free(*pB);
}

using namespace std;

TEMPLATE_TEST_CASE(
    "hipDeviceMalloc()",
    "[device][hipMalloc]",
    (pair<float, plus<>>),
    (pair<float, minus<>>),
    (pair<float, multiplies<>>),
    (pair<float, divides<>>),
    (pair<double, plus<>>),
    (pair<double, minus<>>),
    (pair<double, multiplies<>>),
    (pair<double, divides<>>))
{
    using T = typename TestType::first_type;
    using F = typename TestType::second_type;

    vector<T> A(cnt);
    vector<T> B(cnt);
    vector<T> C(cnt);

    T* Ad;
    T* Bd;
    T* Cd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(T) * cnt) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(T) * cnt) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(T) * cnt) == hipSuccess);

    for (auto i = 0; i != cnt; ++i) {
        A[i] = (i + 1) * T{1.0};
        B[i] = A[i];
        C[i] = A[i];
    }

    REQUIRE(
        hipMemcpy(Ad, data(A), sizeof(T) * size(A), hipMemcpyHostToDevice) ==
        hipSuccess);
    REQUIRE(
        hipMemcpy(Bd, data(B), sizeof(T) * size(B), hipMemcpyHostToDevice) ==
        hipSuccess);

    T** pA;
    T** pB;

    REQUIRE(hipMalloc((void**)&pA, sizeof(T*)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&pB, sizeof(T*)) == hipSuccess);

    dim3 block_dim(block_dim_x, block_dim_y, block_dim_z);
    dim3 grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);

    hipLaunchKernelGGL(
        kernel_alloc<T>, grid_dim, block_dim, 0, 0, Ad, Bd, pA, pB);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    hipLaunchKernelGGL(
        kernel_free<T>, grid_dim, block_dim, 0, 0, pA, pB, Cd, F{});

    REQUIRE(
        hipMemcpy(data(C), Cd, sizeof(T) * size(C), hipMemcpyDeviceToHost) ==
        hipSuccess);

    REQUIRE(hipFree(pA) == hipSuccess);
    REQUIRE(hipFree(pB) == hipSuccess);

    for (auto i = 0; i != cnt; ++i) {
        Approx expected{F{}(A[i], B[i])};

        INFO("i = " << i);
        REQUIRE(C[i] == expected);
    }

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}