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

#include "../../src/include/hip/detail/half.hpp"
#include "../../src/include/hip/detail/helpers.hpp"
#include "hip_vector_types.h"

#include <algorithm>

using __half = hip::detail::half;
using __half2 = hip::detail::half2;

// BEGIN __HALF INTRINSICS
__device__
inline
__half __habs(__half x) noexcept
{
    return abs(x);
}

__device__
inline
__half __hadd(__half x, __half y) noexcept
{
    return x + y;
}

__device__
inline
__half __hadd_sat(__half x, __half y) noexcept
{
    return std::clamp(__hadd(x, y), __half{0}, __half{1});
}

__device__
inline
__half __hdiv(__half x, __half y) noexcept
{
    return x / y;
}

__device__
inline
bool __heq(__half x, __half y) noexcept
{
    return x == y;
}

__device__ std::int32_t __hisnan(__half) noexcept;

__device__
inline
bool __hequ(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __heq(x, y);
}

__device__
inline
__half __hfma(__half x, __half y, __half z) noexcept
{
    return fma(x, y, z);
}

__device__
inline
__half __hfma_sat(__half x, __half y, __half z) noexcept
{
    return std::clamp(__hfma(x, y, z), __half{0}, __half{1});
}

__device__
inline
bool __hge(__half x, __half y) noexcept
{
    return x >= y;
}

__device__
inline
bool __hgeu(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __hge(x, y);
}

__device__
inline
bool __hgt(__half x, __half y) noexcept
{
    return y < x;
}

__device__
inline
bool __hgtu(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __hgt(x, y);
}

__device__
inline
std::int32_t __hisinf(__half x) noexcept
{
    return isinf(x);
}

__device__
inline
std::int32_t __hisnan(__half x) noexcept
{
    return isnan(x);
}

__device__
inline
bool __hle(__half x, __half y) noexcept
{
    return x <= y;
}

__device__
inline
bool __hleu(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __hle(x, y);
}

__device__
inline
bool __hlt(__half x, __half y) noexcept
{
    return x < y;
}

__device__
inline
bool __hltu(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __hlt(x, y);
}

__device__
inline
__half __hmul(__half x, __half y) noexcept
{
    return x * y;
}

__device__
inline
__half __hmul_sat(__half x, __half y) noexcept
{
    return std::clamp(__hmul(x, y), __half{0}, __half{1});
}

__device__
inline
bool __hne(__half x, __half y) noexcept
{
    return x != y;
}

__device__
inline
bool __hneu(__half x, __half y) noexcept
{
    if (__hisnan(x) || __hisnan(y)) return true;
    return __hne(x, y);
}

__device__
inline
__half __hneg(__half x) noexcept
{
    return -x;
}

__device__
inline
__half __hsub(__half x, __half y) noexcept
{
    return x - y;
}

__device__
inline
__half __hsub_sat(__half x, __half y) noexcept
{
    return std::clamp(__hsub(x, y), __half{0}, __half{1});
}
// END __HALF INTRINSICS

// BEGIN __HALF2 INTRINSICS
__device__
inline
__half2 __habs2(__half2 x) noexcept
{
    return __half2{__habs(x.x), __habs(x.y)};
}

__device__
inline
__half2 __hadd2(__half2 x, __half2 y) noexcept
{
    return x + y;
}

__device__
inline
__half2 __hadd2_sat(__half2 x, __half2 y) noexcept
{
    return std::clamp(__hadd2(x, y), __half2{0, 0}, __half2{1, 1});
}

__device__
inline
bool __hbeq2(__half2 x, __half2 y) noexcept
{
    return x == y;
}

__device__
inline
bool __hbequ2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hbeq2(x, y);
}

__device__
inline
bool __hbge2(__half2 x, __half2 y) noexcept
{
    return x >= y;
}

__device__
inline
bool __hbgeu2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hbge2(x, y);
}

__device__
inline
bool __hbgt2(__half2 x, __half2 y) noexcept
{
    return y < x;
}

__device__
inline
bool __hbgtu2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hbgt2(x, y);
}

__device__
inline
bool __hble2(__half2 x, __half2 y) noexcept
{
    return x <= y;
}

__device__
inline
bool __hbleu2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hble2(x, y);
}

__device__
inline
bool __hblt2(__half2 x, __half2 y) noexcept
{
    return x < y;
}

__device__
inline
bool __hbltu2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hblt2(x, y);
}

__device__
inline
bool __hbne2(__half2 x, __half2 y) noexcept
{
    return x != y;
}

__device__
inline
bool __hbneu2(__half2 x, __half2 y) noexcept
{
    if (__hisnan(x.x) || __hisnan(x.y) || __hisnan(y.x) || __hisnan(y.y)) {
        return true;
    }
    return __hbne2(x, y);
}

__device__
inline
__half2 __h2div(__half2 x, __half2 y) noexcept
{
    return x / y;
}

__device__
inline
__half2 __heq2(__half2 x, __half2 y) noexcept
{
    return __half2{__heq(x.x, y.x), __heq(x.y, y.y)};
}

__device__
inline
__half2 __hequ2(__half2 x, __half2 y) noexcept
{
    return __half2{__hequ(x.x, y.x), __hequ(x.y, y.y)};
}

__device__
inline
__half2 __hfma2(__half2 x, __half2 y, __half2 z) noexcept
{
    return x * y + z;
}

__device__
inline
__half2 __hfma2_sat(__half2 x, __half2 y, __half2 z) noexcept
{
    return std::clamp(__hfma2(x, y, z), __half2{0, 0}, __half2{1, 1});
}

__device__
inline
__half2 hge2(__half2 x, __half2 y) noexcept
{
    return __half2{__hge(x.x, y.x), __hge(x.y, y.y)};
}

__device__
inline
__half2 hgeu2(__half2 x, __half2 y) noexcept
{
    return __half2{__hgeu(x.x, y.x), __hgeu(x.y, y.y)};
}

__device__
inline
__half2 hgt2(__half2 x, __half2 y) noexcept
{
    return __half2{__hgt(x.x, y.x), __hgt(x.y, y.y)};
}

__device__
inline
__half2 hgtu2(__half2 x, __half2 y) noexcept
{
    return __half2{__hgtu(x.x, y.x), __hgtu(x.y, y.y)};
}

__device__
inline
__half2 __hisnan2(__half2 x) noexcept
{
    return __half2{__hisnan(x.x), __hisnan(x.y)};
}

__device__
inline
__half2 __hle2(__half2 x, __half2 y) noexcept
{
    return __half2{__hle(x.x, y.x), __hle(x.y, y.y)};
}

__device__
inline
__half2 __hleu2(__half2 x, __half2 y) noexcept
{
    return __half2{__hleu(x.x, y.x), __hleu(x.y, y.y)};
}

__device__
inline
__half2 __hlt2(__half2 x, __half2 y) noexcept
{
    return __half2{__hlt(x.x, y.x), __hlt(x.y, y.y)};
}

__device__
inline
__half2 __hltu2(__half2 x, __half2 y) noexcept
{
    return __half2{__hltu(x.x, y.x), __hltu(x.y, y.y)};
}

__device__
inline
__half2 __hmul2(__half2 x, __half2 y) noexcept
{
    return x * y;
}

__device__
inline
__half2 __hmul2_sat(__half2 x, __half2 y) noexcept
{
    return std::clamp(__hmul2(x, y), __half2{0, 0}, __half2{1, 1});
}

__device__
inline
__half2 __hne2(__half2 x, __half2 y) noexcept
{
    return __half2{__hne(x.x, y.x), __hne(x.y, y.y)};
}

__device__
inline
__half2 __hneu2(__half2 x, __half2 y) noexcept
{
    return __half2{__hneu(x.x, y.x), __hneu(x.y, y.y)};
}

__device__
inline
__half2 __hneg2(__half2 x) noexcept
{
    return -x;
}

__device__
inline
__half2 __hsub2(__half2 x, __half2 y) noexcept
{
    return x - y;
}

__device__
inline
__half2 __hsub2_sat(__half2 x, __half2 y) noexcept
{
    return std::clamp(__hsub2(x, y), __half2{0, 0}, __half2{1, 1});
}
// END __HALF2 INTRINSICS

// BEGIN CONVERSION INTRINSICS
__host__ __device__
inline
__half2 __float22half2_rn(float2 x) noexcept
{
    return __half2{x.x, x.y};
}

__host__ __device__
inline
__half __float2half(float x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
__half2 __float2half2_rn(float x) noexcept
{
    return __half2{x, x};
}

__host__ __device__
inline
__half __float2half_rd(float x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
__half __float2half_rn(float x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
__half __float2half_ru(float x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
__half __float2half_rz(float x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
__half2 __floats2half2_rn(float x, float y) noexcept
{
    return __half2{x, y};
}

__host__ __device__
inline
float2 __half22float2(__half2 x) noexcept
{
    return float2{static_cast<__half>(x.x), static_cast<__half>(x.y)};
}

__host__ __device__
inline
float __half2float(__half x) noexcept
{
    return x;
}

__host__ __device__
inline
__half2 __half2half2(__half x) noexcept
{
    return __half2{x, x};
}

__device__
inline
std::int32_t __half2int_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::int32_t>(x);
}

__device__
inline
std::int32_t __half2int_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::int32_t>(x);
}

__device__
inline
std::int32_t __half2int_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::int32_t>(x);
}

__device__
inline
std::int32_t __half2int_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::int32_t>(x);
}

__device__
inline
std::int64_t __half2ll_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::int64_t>(x);
}

__device__
inline
std::int64_t __half2ll_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::int64_t>(x);
}

__device__
inline
std::int64_t __half2ll_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::int64_t>(x);
}

__device__
inline
std::int64_t __half2ll_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::int64_t>(x);
}

__device__
inline
std::int16_t __half2short_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::int16_t>(x);
}

__device__
inline
std::int16_t __half2short_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::int16_t>(x);
}

__device__
inline
std::int16_t __half2short_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::int16_t>(x);
}

__device__
inline
std::int16_t __half2short_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::int16_t>(x);
}

__device__
inline
std::uint32_t __half2uint_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::uint32_t>(x);
}

__device__
inline
std::uint32_t __half2uint_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::uint32_t>(x);
}

__device__
inline
std::uint32_t __half2uint_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::uint32_t>(x);
}

__device__
inline
std::uint32_t __half2uint_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::uint32_t>(x);
}

__device__
inline
std::uint64_t __half2ull_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::uint64_t>(x);
}

__device__
inline
std::uint64_t __half2ull_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::uint64_t>(x);
}

__device__
inline
std::uint64_t __half2ull_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::uint64_t>(x);
}

__device__
inline
std::uint64_t __half2ull_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::uint64_t>(x);
}

__device__
inline
std::uint16_t __half2ushort_rd(__half x) noexcept
{
    return hip::detail::half_cast<std::uint16_t>(x);
}

__device__
inline
std::uint16_t __half2ushort_rn(__half x) noexcept
{
    return hip::detail::half_cast<std::uint16_t>(x);
}

__device__
inline
std::uint16_t __half2ushort_ru(__half x) noexcept
{
    return hip::detail::half_cast<std::uint16_t>(x);
}

__device__
inline
std::uint16_t __half2ushort_rz(__half x) noexcept
{
    return hip::detail::half_cast<std::uint16_t>(x);
}

__device__
inline
std::int16_t __half_as_short(__half x) noexcept
{
    return hip::detail::bit_cast<std::int16_t>(x);
}

__device__
inline
std::uint16_t __half_as_ushort(__half x) noexcept
{
    return hip::detail::bit_cast<std::uint16_t>(x);
}

__device__
inline
__half2 __halves2half2(__half x, __half y) noexcept
{
    return __half2{x, y};
}

__host__ __device__
inline
float __high2float(__half2 x) noexcept
{
    return hip::detail::half_cast<float>(static_cast<__half>(x.x));
}

__host__ __device__
inline
__half __high2half(__half2 x) noexcept
{
    return x.x;
}

__device__
inline
__half2 __high2half2(__half2 x) noexcept
{
    return __half2{x.x, x.x};
}

__device__
inline
__half2 __highs2half2(__half2 x, __half2 y) noexcept
{
    return __half2{x.x, y.x};
}

__device__
inline
__half __int2half_rd(std::int32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __int2half_rn(std::int32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __int2half_ru(std::int32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __int2half_rz(std::int32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ll2half_rd(std::int64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ll2half_rn(std::int64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ll2half_ru(std::int64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ll2half_rz(std::int64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__host__ __device__
inline
float __low2float(__half2 x) noexcept
{
    return hip::detail::half_cast<float>(static_cast<__half>(x.y));
}

__device__
inline
__half __low2half(__half2 x) noexcept
{
    return x.y;
}

__device__
inline
__half2 __low2half2(__half2 x) noexcept
{
    return __half2{x.y, x.y};
}

__device__
inline
__half2 __lowhigh2highlow(__half2 x) noexcept
{
    return __half2{x.y, x.x};
}

__device__
inline
__half2 __lows2half2(__half2 x, __half2 y) noexcept
{
    return __half2{x.y, y.y};
}

__device__
inline
__half __short2half_rd(std::int16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __short2half_rn(std::int16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __short2half_ru(std::int16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __short2half_rz(std::int16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __short_as_half(std::int16_t x) noexcept
{
    return hip::detail::bit_cast<__half>(x);
}

__device__
inline
__half __uint2half_rd(std::uint32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __uint2half_rn(std::uint32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __uint2half_ru(std::uint32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __uint2half_rz(std::uint32_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ull2half_rd(std::uint64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ull2half_rn(std::uint64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ull2half_ru(std::uint64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ull2half_rz(std::uint64_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ushort2half_rd(std::uint16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ushort2half_rn(std::uint16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ushort2half_ru(std::uint16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ushort2half_rz(std::uint16_t x) noexcept
{
    return hip::detail::half_cast<__half>(x);
}

__device__
inline
__half __ushort_as_half(std::uint16_t x) noexcept
{
    return hip::detail::bit_cast<__half>(x);
}
// END CONVERSION INTRINSICS

// BEGIN HALF MATH FUNCTIONS
inline
__half hcos(__half x) noexcept
{
    return hip::detail::cos(x);
}

inline
__half hexp(__half x) noexcept
{
    return hip::detail::exp(x);
}

inline
__half hlog(__half x) noexcept
{
    return hip::detail::log(x);
}

inline
__half hsin(__half x) noexcept
{
    return hip::detail::sin(x);
}

inline
__half hsqrt(__half x) noexcept
{
    return hip::detail::sqrt(x);
}
// END HALF MATH FUNCTIONS

// BEGIN HALF2 MATH FUNCTIONS
inline
__half2 h2exp(__half2 x) noexcept
{
    return hip::detail::exp(x);
}
// END HALF2 MATH FUNCTIONS