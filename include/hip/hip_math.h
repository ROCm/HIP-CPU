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

#include "../../src/include/hip/detail/math.hpp"

// BEGIN FLOAT INTRINSICS
inline
float __log2f(float x) noexcept
{
    return hip::detail::log2(x);
}

inline
float __powf(float x, float y) noexcept
{
    return hip::detail::pow(x, y);
}
// END FLOAT INTRINSICS

// BEGIN FLOAT FUNCTIONS
inline
float erfcinvf(float x) noexcept
{
    return hip::detail::erfcinv(x);
}

inline
float erfinvf(float x) noexcept
{
    return hip::detail::erfinv(x);
}

inline
float erfcxf(float x) noexcept
{
    return hip::detail::erfcx(x);
}

inline
float fdividef(float x, float y) noexcept
{
    return hip::detail::fdivide(x, y);
}

inline
float normf(std::int32_t dim, const float* p) noexcept
{
    return hip::detail::norm(dim, p);
}

inline
float normcdff(float x) noexcept
{
    return hip::detail::normcdf(x);
}

inline
float normcdfinvf(float x) noexcept
{
    return hip::detail::normcdfinv(x);
}

inline
float norm3df(float x, float y, float z) noexcept
{
    return hip::detail::norm3d(x, y, z);
}

inline
float norm4df(float x, float y, float z, float w) noexcept
{
    return hip::detail::norm4d(x, y, z, w);
}

inline
float rcbrtf(float x) noexcept
{
    return hip::detail::rcbrt(x);
}

inline
float rhypotf(float x, float y) noexcept
{
    return hip::detail::rhypot(x, y);
}

inline
float rnormf(std::int32_t dim, const float* p) noexcept
{
    return hip::detail::rnorm(dim, p);
}

inline
float rnorm3df(float x, float y, float z) noexcept
{
    return hip::detail::rnorm3d(x, y, z);
}

inline
float rnorm4df(float x, float y, float z, float w) noexcept
{
    return hip::detail::rnorm4d(x, y, z, w);
}

inline
void sincosf(float x, float* psin, float* pcos) noexcept
{
    return hip::detail::sincos(x, psin, pcos);
}

inline
void sincospif(float x, float* psin, float* pcos) noexcept
{
    return hip::detail::sincospi(x, psin, pcos);
}
// END FLOAT FUNCTIONS

// BEGIN DOUBLE FUNCTIONS
inline
double erfcinv(double x) noexcept
{
    return hip::detail::erfcinv(x);
}

inline
double erfinv(double x) noexcept
{
    return hip::detail::erfinv(x);
}

inline
double erfcx(double x) noexcept
{
    return hip::detail::erfcx(x);
}

inline
double fdivide(double x, double y) noexcept
{
    return hip::detail::fdivide(x, y);
}

inline
double normcdf(double x) noexcept
{
    return hip::detail::normcdf(x);
}

inline
double normcdfinv(double x) noexcept
{
    return hip::detail::normcdfinv(x);
}

inline
double norm3d(double x, double y, double z) noexcept
{
    return hip::detail::norm3d(x, y, z);
}

inline
double norm4d(double x, double y, double z, double w) noexcept
{
    return hip::detail::norm4d(x, y, z, w);
}

inline
double rcbrt(double x) noexcept
{
    return hip::detail::rcbrt(x);
}

inline
double rhypot(double x, double y) noexcept
{
    return hip::detail::rhypot(x, y);
}

inline
double rnorm(std::int32_t dim, const double* p) noexcept
{
    return hip::detail::rnorm(dim, p);
}

inline
double rnorm3d(double x, double y, double z) noexcept
{
    return hip::detail::rnorm3d(x, y, z);
}

inline
double rnorm4d(double x, double y, double z, double w) noexcept
{
    return hip::detail::rnorm4d(x, y, z, w);
}

inline
void sincos(double x, double* psin, double* pcos) noexcept
{
    return hip::detail::sincos(x, psin, pcos);
}

inline
void sincospi(double x, double* psin, double* pcos) noexcept
{
    return hip::detail::sincospi(x, psin, pcos);
}
// END DOUBLE FUNCTIONS

// BEGIN INTEGER INTRINSICS
inline
std::int32_t __mul24(std::int32_t x, std::int32_t y) noexcept
{
    return hip::detail::mul24(x, y);
}
// END INTEGER INTRINSICS