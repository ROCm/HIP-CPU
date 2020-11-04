/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <cstdlib>

TEST_CASE("hipDeviceGetLimit()", "[host][hipDevice]")
{
    constexpr auto current_HIP_heap_sz{4194304u};
    std::size_t heap_sz{};

    REQUIRE(hipDeviceGetLimit(&heap_sz, hipLimitMallocHeapSize) == hipSuccess);
    REQUIRE(heap_sz >= current_HIP_heap_sz);
}