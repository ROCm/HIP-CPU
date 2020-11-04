/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

TEST_CASE("hipSetDeviceFlags()", "[host][hipDevice]")
{
    REQUIRE(hipDeviceReset() == hipSuccess);

    int deviceCount = 0;
    REQUIRE(hipGetDeviceCount(&deviceCount) == hipSuccess);

    for (int i = 0; i != deviceCount; ++i) {
        REQUIRE(hipSetDevice(i) == hipSuccess);

        int flag = 0;
        for (auto flag = 1; flag != hipDeviceLmemResizeToMax; flag <<= 1) {
            INFO("Flag = " << flag);
            REQUIRE(hipSetDeviceFlags(flag) == hipSuccess);
        }
    }
}