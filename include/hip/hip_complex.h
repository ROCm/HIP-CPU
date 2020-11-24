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

#include "../../src/include/hip/detail/complex.hpp"

using hipFloatComplex = hip::detail::complex<float>;
using hipComplex = hipFloatComplex;
using hipDoubleComplex = hip::detail::complex<double>;

// BEGIN HIPFLOATCOMPLEX FUNCTIONS
inline
float hipCabsf(hipFloatComplex x) noexcept
{
    return hip::detail::abs(x);
}

inline
hipFloatComplex hipCaddf(hipFloatComplex x, hipFloatComplex y) noexcept
{
    return x + y;
}

inline
hipFloatComplex hipCdivf(hipFloatComplex x, hipFloatComplex y) noexcept
{
    return hip::detail::divide(x, y);
}

inline
float hipCimagf(hipFloatComplex x) noexcept
{
    return hip::detail::imag(x);
}

inline
hipFloatComplex hipCmulf(hipFloatComplex x, hipFloatComplex y) noexcept
{
    return hip::detail::multiply(x, y);
}

inline
hipFloatComplex hipConjf(hipFloatComplex x) noexcept
{
    return hip::detail::conjugate(x);
}

inline
float hipCrealf(hipFloatComplex x) noexcept
{
    return hip::detail::real(x);
}

inline
float hipCsqabsf(hipFloatComplex x) noexcept
{
    return hip::detail::absolute_square(x);
}

inline
hipFloatComplex hipCsubf(hipFloatComplex x, hipFloatComplex y) noexcept
{
    return x - y;
}
// END HIPFLOATCOMPLEX FUNCTIONS

// BEGIN HIPDOUBLECOMPLEX FUNCTIONS
// END HIPDOUBLECOMPLEX FUNCTIONS