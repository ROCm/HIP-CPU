/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

__global__
void run_printf() { printf("Hello World\n"); }

TEST_CASE("printf()", "[device][printf]")
{
    int device_count = 0;
    REQUIRE(hipGetDeviceCount(&device_count) == hipSuccess);

    for (int i = 0; i != device_count; ++i) {
        REQUIRE(hipSetDevice(i) == hipSuccess);

        hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);
    }
}