/* -----------------------------------------------------------------------------
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>

using namespace std;

TEST_CASE("Unit_hipMallocAsync_negative", "[host][malloc][async]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    void* p{};
    const auto max_size{std::numeric_limits<size_t>::max()};
    hipStream_t stream{};

    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    SECTION("Device pointer is null") {
        REQUIRE(hipMallocAsync(nullptr, 100, stream) != hipSuccess);
    }

    SECTION("stream is invalid")
    {
        REQUIRE(hipMallocAsync(
            static_cast<void**>(&p),
            100,
            reinterpret_cast<hipStream_t>(-1)) != hipSuccess);
    }

    SECTION("out of memory") {
        REQUIRE(hipMallocAsync(
            static_cast<void**>(&p), max_size, stream) != hipSuccess);
    }

    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE("Unit_hipMalloc_Positive_Zero_Size", "[host][malloc]")
{
    void* ptr{reinterpret_cast<void*>(0x1)};

    REQUIRE(hipMalloc(&ptr, 0) == hipSuccess);

    REQUIRE(ptr == nullptr);
}

TEST_CASE("Unit_hipMalloc_Positive_Alignment", "[host][malloc]")
{
  void* ptr1{};
  void* ptr2{};

  REQUIRE(hipMalloc(&ptr1, 1) == hipSuccess);
  REQUIRE(hipMalloc(&ptr2, 10) == hipSuccess);

  CHECK(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
  CHECK(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);

  REQUIRE(hipFree(ptr1) == hipSuccess);
  REQUIRE(hipFree(ptr2) == hipSuccess);
}

TEST_CASE("Unit_hipMalloc_Negative_Parameters", "[host][malloc]")
{
    SECTION("ptr == nullptr") {
        REQUIRE(hipMalloc(nullptr, 4096) == hipErrorInvalidValue);
    }
    SECTION("size == max size_t") {
        void* ptr{};
        REQUIRE(hipMalloc(
            &ptr, std::numeric_limits<size_t>::max()) == hipErrorOutOfMemory);
    }
}

__global__ void delay_kernel(atomic<bool>* done) { while (!*done); }

TEST_CASE("Unit_hipFreeImplicitSyncDev", "[host][free]")
{
    int* devPtr{};
    const auto size_mult{GENERATE(1, 32, 64, 128, 256)};

    REQUIRE(hipMalloc(&devPtr, sizeof(*devPtr) * size_mult) == hipSuccess);

    atomic<bool> done{false};
    hipLaunchKernelGGL(delay_kernel, dim3{1}, dim3{1}, 0, nullptr, &done);

    REQUIRE(hipStreamQuery(nullptr) == hipErrorNotReady);

    done = true;

    REQUIRE(hipFree(devPtr) == hipSuccess);
    REQUIRE(hipStreamQuery(nullptr) == hipSuccess);
}

TEST_CASE("Unit_hipFreeImplicitSyncHost", "[host][free]")
{
    int* hostPtr{};
    const auto size_mult{GENERATE(1, 32, 64, 128, 256)};

    REQUIRE(
        hipHostMalloc(&hostPtr, sizeof(*hostPtr) * size_mult) == hipSuccess);

    atomic<bool> done{false};
    hipLaunchKernelGGL(delay_kernel, dim3{1}, dim3{1}, 0, nullptr, &done);

    REQUIRE(hipStreamQuery(nullptr) == hipErrorNotReady);

    done = true;

    REQUIRE(hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hipStreamQuery(nullptr) == hipSuccess);
}

constexpr size_t numAllocs{10};

TEMPLATE_TEST_CASE(
    "Unit_hipFreeMultiTDev", "[host][free]", char, int, float2, float4)
{
    vector<TestType*> ptrs{numAllocs};
    auto allocSize{sizeof(TestType) * GENERATE(1, 32, 64, 128)};

    for (auto&& ptr : ptrs) REQUIRE(hipMalloc(&ptr, allocSize) == hipSuccess);

    vector<thread> threads;

    for (auto&& ptr : ptrs) {
        threads.emplace_back(([=]() {
            REQUIRE(hipFree(ptr) == hipSuccess);
            REQUIRE(hipStreamQuery(nullptr) == hipSuccess);
        }));
    }

    for (auto&& t : threads) t.join();
}

TEMPLATE_TEST_CASE(
    "Unit_hipFreeMultiTHost", "[host][free]", char, int, float2, float4)
{
    vector<TestType*> ptrs{numAllocs};
    auto allocSize{sizeof(TestType) * GENERATE(1, 32, 64, 128)};

    for (auto&& ptr : ptrs) {
        REQUIRE(hipHostMalloc(&ptr, allocSize) == hipSuccess);
    }

    vector<thread> threads;

    for (auto&& ptr : ptrs) {
        threads.emplace_back(([=]() {
            REQUIRE(hipHostFree(ptr) == hipSuccess);
            REQUIRE(hipStreamQuery(nullptr) == hipSuccess);
        }));
    }

    for (auto&& t : threads) t.join();
}

TEST_CASE("Unit_hipFreeAsync_negative", "[host][free][async]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    void* p{};

    hipStream_t stream{};
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    SECTION("dev_ptr is nullptr") {
        REQUIRE(hipFreeAsync(nullptr, stream) != hipSuccess);
    }

    SECTION("invalid stream handle") {
        REQUIRE(
            hipMallocAsync(static_cast<void**>(&p), 100, stream) == hipSuccess);
        REQUIRE(hipStreamSynchronize(stream) == hipSuccess);

        REQUIRE(hipFreeAsync(p, stream) == hipSuccess);
        REQUIRE(hipStreamSynchronize(stream) == hipSuccess);
    }

    REQUIRE(hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE("Unit_hipMalloc_Positive_Basic", "[host][malloc]")
{
    constexpr size_t page_size{4096};
    void* ptr{};
    const auto alloc_size{GENERATE_COPY(
        10, page_size / 2, page_size, page_size * 3 / 2, page_size * 2)};

    REQUIRE(hipMalloc(&ptr, alloc_size) == hipSuccess);

    CHECK(ptr != nullptr);
    CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);

    REQUIRE(hipFree(ptr) == hipSuccess);
}