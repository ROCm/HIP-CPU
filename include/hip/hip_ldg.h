#pragma once

#include <version>
#if !defined(__cpp_lib_parallel_algorithm)
    #error The HIP-CPU RT requires a C++17 compliant standard library which exposes parallel algorithms support; see https://en.cppreference.com/w/cpp/algorithm#Execution_policies.
#endif

#define __HIP_CPU_RT__

#include "hip_vector_types.h"

inline char __ldg(const char *ptr) noexcept { return *ptr; }
inline signed char __ldg(const signed char *ptr) noexcept { return *ptr; }
inline short __ldg(const short *ptr) noexcept { return *ptr; }
inline int __ldg(const int *ptr) noexcept { return *ptr; }
inline long __ldg(const long *ptr) noexcept { return *ptr; }
inline long long __ldg(const long long *ptr) noexcept { return *ptr; }

inline char2 __ldg(const char2 *ptr) noexcept { return *ptr; }
inline char4 __ldg(const char4 *ptr) noexcept { return *ptr; }
inline short2 __ldg(const short2 *ptr) noexcept { return *ptr; }
inline short4 __ldg(const short4 *ptr) noexcept { return *ptr; }
inline int2 __ldg(const int2 *ptr) noexcept { return *ptr; }
inline int4 __ldg(const int4 *ptr) noexcept { return *ptr; }
inline longlong2 __ldg(const longlong2 *ptr) noexcept { return *ptr; }

inline unsigned char __ldg(const unsigned char *ptr) noexcept { return *ptr; }
inline unsigned short __ldg(const unsigned short *ptr) noexcept { return *ptr; }
inline unsigned int __ldg(const unsigned int *ptr) noexcept { return *ptr; }
inline unsigned long __ldg(const unsigned long *ptr) noexcept { return *ptr; }
inline unsigned long long __ldg(const unsigned long long *ptr) noexcept { return *ptr; }

inline uchar2 __ldg(const uchar2 *ptr) noexcept { return *ptr; }
inline uchar4 __ldg(const uchar4 *ptr) noexcept { return *ptr; }
inline ushort2 __ldg(const ushort2 *ptr) noexcept { return *ptr; }
inline ushort4 __ldg(const ushort4 *ptr) noexcept { return *ptr; }
inline uint2 __ldg(const uint2 *ptr) noexcept { return *ptr; }
inline uint4 __ldg(const uint4 *ptr) noexcept { return *ptr; }
inline ulonglong2 __ldg(const ulonglong2 *ptr) noexcept { return *ptr; }

inline float __ldg(const float *ptr) noexcept { return *ptr; }
inline double __ldg(const double *ptr) noexcept { return *ptr; }

inline float2 __ldg(const float2 *ptr) noexcept { return *ptr; }
inline float4 __ldg(const float4 *ptr) noexcept { return *ptr; }
inline double2 __ldg(const double2 *ptr) noexcept { return *ptr; }
