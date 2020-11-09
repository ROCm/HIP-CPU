/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <thread>
#include <vector>

using namespace std;

constexpr auto cnt{10u * 10u};

TEST_CASE(
    "hipStream_t creation & destruction", "[host][multithread][hipStream_t]")
{
    vector<hipStream_t> streams(cnt);

    for (decltype(size(streams)) i = 0u; i != sqrt(size(streams)); ++i) {
        for (decltype(size(streams)) j = 0u; j != sqrt(size(streams)); ++j) {
            REQUIRE(hipStreamCreate(&streams[j]) == hipSuccess);
        }
        for (decltype(size(streams)) j = 0u; j != sqrt(size(streams)); ++j) {
            REQUIRE(hipStreamDestroy(streams[j]) == hipSuccess);
        }
    }
}

TEST_CASE(
    "Serialised multi-threaded hipStream_t creation & destruction",
    "[host][multithread][hipStream_t]")
{
    thread{[]() {
        vector<hipStream_t> streams(100);
        for (auto i = 0u; i != 3; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }}.join();

    thread{[]() {
        vector<hipStream_t> streams(10);
        for (auto i = 0u; i != 30; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }}.join();

    thread{[]() {
        vector<hipStream_t> streams(1);
        for (auto i = 0u; i != 300; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }}.join();
}

TEST_CASE(
    "Parallel, asymmetric, multi-threaded hipStream_t creation & destruction",
    "[host][multithread][hipStream_t]")
{
    thread t0{[]() {
        vector<hipStream_t> streams(100);
        for (auto i = 0u; i != 3; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t1{[]() {
        vector<hipStream_t> streams(10);
        for (auto i = 0u; i != 30; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t2{[]() {
        vector<hipStream_t> streams(1);
        for (auto i = 0u; i != 300; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());
    REQUIRE_NOTHROW(t2.join());
}

TEST_CASE(
    "Parallel, asymmetric, multi-threaded hipStream_t creation & destruction, "
    "many streams",
    "[host][multithread][hipStream_t]")
{
    thread t0{[]() {
        vector<hipStream_t> streams(100);
        for (auto i = 0u; i != 100; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t1{[]() {
        vector<hipStream_t> streams(10);
        for (auto i = 0u; i != 1000; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t2{[]() {
        vector<hipStream_t> streams(1);
        for (auto i = 0u; i != 10000; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());
    REQUIRE_NOTHROW(t2.join());
}

TEST_CASE(
    "Parallel, asymmetric, multi-threaded hipStream_t creation & destruction, "
    "many streams, paired creation & destruction",
    "[host][multithread][hipStream_t]")
{
    thread t0{[]() {
        vector<hipStream_t> streams(1);
        for (auto i = 0u; i != 1000; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t1{[]() {
        vector<hipStream_t> streams(1);
        for (auto i = 0u; i != 1000; ++i) {
            for (auto&& x : streams) REQUIRE(hipStreamCreate(&x) == hipSuccess);
            for (auto&& x : streams) REQUIRE(hipStreamDestroy(x) == hipSuccess);
        }
    }};

    thread t2{[]() {
        for (auto i = 0u; i != 50000; ++i) {
            REQUIRE(hipDeviceSynchronize() == hipSuccess);
        }
    }};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());
    REQUIRE_NOTHROW(t2.join());
}