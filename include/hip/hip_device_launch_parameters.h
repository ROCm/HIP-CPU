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

#include "../../src/include/hip/detail/coordinates.hpp"

inline constexpr hip::detail::Block_dim blockDim{};
inline constexpr hip::detail::Block_idx blockIdx{};
inline constexpr hip::detail::Grid_dim gridDim{};
inline constexpr hip::detail::Thread_idx threadIdx{};