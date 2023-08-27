/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <bitset>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

constexpr auto WIDTH{8};
constexpr auto HEIGHT{8};
constexpr auto NUM{WIDTH * HEIGHT};

constexpr auto THREADS_PER_BLOCK_X{8};
constexpr auto THREADS_PER_BLOCK_Y{8};
constexpr auto THREADS_PER_BLOCK_Z{1};

using namespace std;

template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
uint32_t firstbit(T x)
{
    if (x == 0) return sizeof(T) * CHAR_BIT;

    bitset<sizeof(T) * CHAR_BIT> tmp(x);

    auto n{size(tmp)};
    while (n--) {
        if (tmp[n]) return static_cast<std::uint32_t>(size(tmp) - n - 1);
    }

    return static_cast<std::uint32_t>(size(tmp));
}

namespace
{
    __global__
    void HIP_kernel(
        uint32_t* a,
        uint32_t* b,
        uint32_t* c,
        uint64_t* d,
        uint32_t width,
        uint32_t height)
    {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto y = blockDim.y * blockIdx.y + threadIdx.y;

        const auto i = y * width + x;
        if (i < (width * height)) {
            a[i] = __clz(b[i]);
            c[i] = __clzll(d[i]);
        }
    }
} // Unnamed namespace.

TEST_CASE("clz()", "[device][clz]")
{
    vector<uint32_t> hostA(NUM);
    vector<uint32_t> hostB(NUM);
    vector<uint32_t> hostC(NUM);
    vector<uint64_t> hostD(NUM);

    generate_n(begin(hostB), size(hostB), [i = 0u]() mutable {
        return 419430 * i++;
    });
    generate_n(begin(hostD), size(hostD), [i = 0u]() mutable { return i++; });

    uint32_t* deviceA;
    uint32_t* deviceB;
    uint32_t* deviceC;
    uint64_t* deviceD;

    REQUIRE(hipMalloc(&deviceA, NUM * sizeof(uint32_t)) == hipSuccess);
    REQUIRE(hipMalloc(&deviceB, NUM * sizeof(uint32_t)) == hipSuccess);
    REQUIRE(hipMalloc(&deviceC, NUM * sizeof(uint32_t)) == hipSuccess);
    REQUIRE(hipMalloc(&deviceD, NUM * sizeof(uint64_t)) == hipSuccess);

    REQUIRE(hipMemcpy(
        deviceB,
        data(hostB),
        NUM * sizeof(uint32_t),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        deviceD,
        data(hostD),
        NUM * sizeof(uint64_t),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        HIP_kernel,
        dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
        0,
        0,
        deviceA,
        deviceB,
        deviceC,
        deviceD,
        WIDTH,
        HEIGHT);

    REQUIRE(hipMemcpy(
        data(hostA),
        deviceA,
        NUM * sizeof(uint32_t),
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        data(hostC),
        deviceC,
        NUM * sizeof(unsigned int),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(
        equal(cbegin(hostA), cend(hostA), cbegin(hostB), [](auto x, auto y) {
        return x == firstbit(y);
    }));
    REQUIRE(
        equal(cbegin(hostC), cend(hostC), cbegin(hostD), [](auto x, auto y) {
        return x == firstbit(y);
    }));

    REQUIRE(hipFree(deviceA) == hipSuccess);
    REQUIRE(hipFree(deviceB) == hipSuccess);
    REQUIRE(hipFree(deviceC) == hipSuccess);
    REQUIRE(hipFree(deviceD) == hipSuccess);
}