/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

using namespace std;

constexpr auto cnt{1024};
constexpr auto byte_cnt{cnt * sizeof(uint32_t)};

__global__
void cpy(uint32_t* Out, uint32_t* In)
{
    int tx = threadIdx.x;
    memcpy(Out + tx, In + tx, sizeof(uint32_t));
}

__global__
void set(uint32_t* ptr, uint8_t val, size_t size)
{
    int tx = threadIdx.x;
    memset(ptr + tx, val, sizeof(uint32_t));
}

TEST_CASE("hipDeviceMemset()", "[device][hipMemset]")
{
    vector<uint32_t> A(cnt);
    vector<uint32_t> B(cnt, 0u);

    iota(begin(A), end(A), 0u);

    uint32_t* Ad;
    uint32_t* Bd;
    REQUIRE(hipMalloc((void**)&Ad, byte_cnt) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, byte_cnt) == hipSuccess);

    REQUIRE(
        hipMemcpy(Ad, data(A), byte_cnt, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(cpy, dim3(1), dim3(cnt), 0, 0, Bd, Ad);

    REQUIRE(
        hipMemcpy(data(B), Bd, byte_cnt, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(A == B);

    hipLaunchKernelGGL(::set, dim3(1), dim3(cnt), 0, 0, Bd, 0x1, cnt);

    REQUIRE(
        hipMemcpy(data(B), Bd, byte_cnt, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(
        all_of(cbegin(B), cend(B), [](auto&& x) { return x == 0x01010101; }));
}