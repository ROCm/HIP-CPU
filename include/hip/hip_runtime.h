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

#include "hip_api.h"
#include "hip_atomic.h"
#include "hip_complex.h"
#include "hip_constants.h"
#include "hip_defines.h"
#include "hip_device_launch_parameters.h"
#include "hip_enums.h"
#include "hip_fp16.h"
#include "hip_ldg.h"
#include "hip_math.h"
#include "hip_types.h"
#include "hip_vector_types.h"
