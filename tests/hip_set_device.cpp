/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

TEST_CASE("hipSetDevice()", "[host][hipDevice][set_device]")
{
    int numDevices = 0;

    REQUIRE(hipGetDeviceCount(&numDevices) == hipSuccess);
    REQUIRE(numDevices == 1); // TODO: Temporary HIP-CPU specific.

    for (auto i = 0; i != numDevices; ++i) {
        REQUIRE(hipSetDevice(i) == hipSuccess);
    }

    REQUIRE(hipSetDevice(numDevices) == hipErrorInvalidDevice);
}