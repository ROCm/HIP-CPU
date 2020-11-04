/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#include <version>
#if !defined(__cpp_lib_parallel_algorithm)
    #error The HIP-CPU RT requires a C++17 compliant standard library which exposes parallel algorithms support; see https://en.cppreference.com/w/cpp/algorithm#Execution_policies.
#endif

#define __HIP_CPU_RT__

#include <cstdint>

namespace hip
{
    namespace constants
    {
        inline constexpr auto hipCPUDeviceID{INT32_MAX};
    } // Namespace hip::constants.
} // Namespace hip.

inline constexpr int warpSize{64}; // TODO: this is untrue and a problem.

inline const auto HIP_LAUNCH_PARAM_BUFFER_POINTER{reinterpret_cast<void*>(1)};
inline const auto HIP_LAUNCH_PARAM_BUFFER_SIZE{reinterpret_cast<void*>(2)};
inline const auto HIP_LAUNCH_PARAM_END{reinterpret_cast<void*>(3)};