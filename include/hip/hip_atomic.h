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

#if defined(_MSC_VER) && !defined(__clang__)
    #include "../../src/include/hip/detail/atomic_msvc.hpp"
#else
    #include "../../src/include/hip/detail/atomic_clang_gcc.hpp"
#endif

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// BEGIN ADD
inline
int atomicAdd(int* address, int val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
unsigned int atomicAdd(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
unsigned long atomicAdd(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
unsigned long long atomicAdd(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
float atomicAdd(float* address, float val) noexcept
{
    return hip::detail::atomic_cas_add(address, val);
}

inline
double atomicAdd(double* address, double val) noexcept
{
    return hip::detail::atomic_cas_add(address, val);
}
// TODO: add atomics after introducing half.
// __half2 atomicAdd(__half2 *address, __half2 val);
// __half atomicAdd(__half *address, __half val);
// END ADD

// BEGIN AND
inline
int atomicAnd(int* address, int val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

inline
unsigned int atomicAnd(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

inline
unsigned long atomicAnd(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

// FIXME: available in CUDA, not in HIP/ROCm
inline
long long atomicAnd(long long* address, long long val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

inline
unsigned long long atomicAnd(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_and(address, val);
}
// END AND

// BEGIN CAS
// FIXME: not available in CUDA or in HIP/ROCm
inline
unsigned short atomicCAS(
    unsigned short* address,
    unsigned short compare,
    unsigned short val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
int atomicCAS(
    int* address, int compare, int val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
unsigned int atomicCAS(
    unsigned int* address, unsigned int compare, unsigned int val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
unsigned long atomicCAS(
    unsigned long* address, unsigned long compare, unsigned long val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
unsigned long long atomicCAS(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}
// FIXME: HIP/ROCm supports also atomicCAS with float and double arguments
// END CAS

// BEGIN DECREMENT
// FIXME: overflow/underflow handling is missing
inline
unsigned int atomicDec(unsigned int* address, unsigned int /*val*/) noexcept
{
    return hip::detail::atomic_dec(address);
}
// END DECREMENT

// BEGIN EXCHANGE
inline
int atomicExch(int* address, int val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
unsigned int atomicExch(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
unsigned long atomicExch(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
unsigned long long atomicExch(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
float atomicExch(float* address, float val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}
// FIXME: HIP/ROCm supports also atomicExch with double arguments
// END EXCHANGE

// BEGIN INCREMENT
// FIXME: overflow handling is missing
inline
unsigned int atomicInc(unsigned int* address, unsigned int /*val*/) noexcept
{
    return hip::detail::atomic_inc(address);
}
// END INCREMENT

// BEGIN MAX
inline
int atomicMax(int* address, int val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
unsigned int atomicMax(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
unsigned long atomicMax(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
long long atomicMax(long long* address, long long val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_max(address, val);
}
// FIXME: HIP/ROCm supports also atomicMax with float and double arguments
// END MAX

// BEGIN MIN
inline
int atomicMin(int* address, int val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
unsigned int atomicMin(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
unsigned long atomicMin(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
long long atomicMin(long long* address, long long val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_min(address, val);
}
// FIXME: HIP/ROCm supports also atomicMin with float and double arguments
// END MIN

// BEGIN OR
inline
int atomicOr(int* address, int val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

inline
unsigned int atomicOr(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

inline
unsigned long atomicOr(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

// FIXME: available in CUDA, not in HIP/ROCm
inline
long long atomicOr(long long* address, long long val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

inline
unsigned long long atomicOr(unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_or(address, val);
}
// END OR

// BEGIN SUB
inline
int atomicSub(int* address, int val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}

inline
unsigned int atomicSub(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}

inline
unsigned long atomicSub(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}

inline
unsigned long long atomicSub(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}
// FIXME: HIP/ROCm supports also atomicSub with float and double arguments
// END SUB

// BEGIN XOR
inline
int atomicXor(int* address, int val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

inline
unsigned int atomicXor(unsigned int* address, unsigned int val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

inline
unsigned long atomicXor(unsigned long* address, unsigned long val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

// FIXME: available in CUDA, not in HIP/ROCm
inline
long long atomicXor(long long* address, long long val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

inline
unsigned long long atomicXor(
    unsigned long long* address, unsigned long long val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}
// END XOR
