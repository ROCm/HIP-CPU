/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cfloat>

TEST_CASE("Negative tests.", "[host][hipEvent_t][elapsed_time]")
{
    SECTION("Null pointers.") {
        hipEvent_t start{};
        hipEvent_t end{};

        float tms{FLT_MAX};
        REQUIRE(
            hipEventElapsedTime(nullptr, start, end) == hipErrorInvalidValue);

        REQUIRE(
            hipEventElapsedTime(&tms, nullptr, end) == hipErrorInvalidHandle);
        REQUIRE(
            hipEventElapsedTime(&tms, start, nullptr) == hipErrorInvalidHandle);
    }

    SECTION("Disabled timing.") {
        hipEvent_t start;
        hipEvent_t stop;
        REQUIRE(hipEventCreateWithFlags(
            &start, hipEventDisableTiming) == hipSuccess);
        REQUIRE(hipEventCreateWithFlags(
            &stop, hipEventDisableTiming) == hipSuccess);

        float timeElapsed{FLT_MAX};
        REQUIRE(hipEventElapsedTime(
            &timeElapsed, start, stop) == hipErrorInvalidHandle);
    }
}

TEST_CASE("Positive tests.", "[host][hipEvent_t][elapsed_time]")
{
    hipEvent_t start;
    REQUIRE(hipEventCreate(&start) == hipSuccess);

    hipEvent_t stop;
    REQUIRE(hipEventCreate(&stop) == hipSuccess);

    REQUIRE(hipEventRecord(start, nullptr) == hipSuccess);
    REQUIRE(hipEventSynchronize(start) == hipSuccess);

    REQUIRE(hipEventRecord(stop, nullptr) == hipSuccess);
    REQUIRE(hipEventSynchronize(stop) == hipSuccess);

    float tElapsed{FLT_MAX};
    REQUIRE(hipEventElapsedTime(&tElapsed, start, stop) == hipSuccess);
    REQUIRE(tElapsed != Approx{FLT_MAX});
}