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
std::int32_t atomicAdd(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
std::uint32_t atomicAdd(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

inline
std::uint64_t atomicAdd(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_add(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
std::uint64_t atomicAdd(T* address, T val) noexcept
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
std::int32_t atomicAnd(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

inline
std::uint32_t atomicAnd(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

inline
std::uint64_t atomicAnd(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_and(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicAnd(T* address, T val) noexcept
{
    return hip::detail::atomic_and(address, val);
}
// END AND

// BEGIN CAS
inline
std::uint16_t atomicCAS(
    std::uint16_t* address, std::uint16_t compare, std::uint16_t val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
std::int32_t atomicCAS(
    std::int32_t* address, std::int32_t compare, std::int32_t val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
std::uint32_t atomicCAS(
    std::uint32_t* address, std::uint32_t compare, std::uint32_t val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

inline
std::uint64_t atomicCAS(
    std::uint64_t* address, std::uint64_t compare, std::uint64_t val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicCAS(T* address, T compare, T val) noexcept
{
    return hip::detail::atomic_cas(address, compare, val);
}
// END CAS

// BEGIN DECREMENT
inline
std::uint32_t atomicDec(std::uint32_t* address, std::uint32_t /*val*/) noexcept
{
    return hip::detail::atomic_dec(address);
}
// END DECREMENT

// BEGIN EXCHANGE
inline
std::int32_t atomicExch(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
std::uint32_t atomicExch(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
std::uint64_t atomicExch(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicExch(T* address, T val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}

inline
float atomicExch(float* address, float val) noexcept
{
    return hip::detail::atomic_exchange(address, val);
}
// END EXCHANGE

// BEGIN INCREMENT
inline
std::uint32_t atomicInc(std::uint32_t* address, std::uint32_t /*val*/) noexcept
{
    return hip::detail::atomic_inc(address);
}
// END INCREMENT

// BEGIN MAX
inline
std::int32_t atomicMax(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
std::uint32_t atomicMax(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

inline
std::uint64_t atomicMax(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_max(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicMax(T* address, T val) noexcept
{
    return hip::detail::atomic_max(address, val);
}
// END MAX

// BEGIN MIN
inline
std::int32_t atomicMin(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
std::uint32_t atomicMin(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

inline
std::uint64_t atomicMin(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_min(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicMin(T* address, T val) noexcept
{
    return hip::detail::atomic_min(address, val);
}
// END MIN

// BEGIN OR
inline
std::int32_t atomicOr(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

inline
std::uint32_t atomicOr(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

inline
std::uint64_t atomicOr(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_or(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicOr(T* address, T val) noexcept
{
    return hip::detail::atomic_or(address, val);
}
// END OR

// BEGIN SUB
inline
std::int32_t atomicSub(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}

inline
std::uint32_t atomicSub(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_sub(address, val);
}
// END SUB

// BEGIN XOR
inline
std::int32_t atomicXor(std::int32_t* address, std::int32_t val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

inline
std::uint32_t atomicXor(std::uint32_t* address, std::uint32_t val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

inline
std::uint64_t atomicXor(std::uint64_t* address, std::uint64_t val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}

template<
    typename T,
    std::enable_if_t<
        std::is_same_v<T, unsigned long long> &&
        !std::is_same_v<unsigned long long, std::uint64_t>>* = nullptr>
inline
T atomicXor(T* address, T val) noexcept
{
    return hip::detail::atomic_xor(address, val);
}
// END XOR