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

#include "../../src/include/hip/detail/vector.hpp"

#define MAKE_VECTOR_TYPE(type, name)\
    using name##1 = hip::detail::Vector_type<type, 1>;\
    using name##2 = hip::detail::Vector_type<type, 2>;\
    using name##3 = hip::detail::Vector_type<type, 3>;\
    using name##4 = hip::detail::Vector_type<type, 4>;\
\
    inline\
    name##1 make_##name##1(type x) noexcept\
    {\
        return name##1{x};\
    }\
    inline\
    name##2 make_##name##2(type x, type y) noexcept\
    {\
        return name##2{x, y};\
    }\
    inline\
    name##3 make_##name##3(type x, type y, type z) noexcept\
    {\
        return name##3{x, y, z};\
    }\
    inline\
    name##4 make_##name##4(type x, type y, type z, type w) noexcept\
    {\
        return name##4{x, y, z, w};\
    }

MAKE_VECTOR_TYPE(char, char)
MAKE_VECTOR_TYPE(unsigned char, uchar)
MAKE_VECTOR_TYPE(short, short)
MAKE_VECTOR_TYPE(unsigned short, ushort)
MAKE_VECTOR_TYPE(int, int)
MAKE_VECTOR_TYPE(unsigned int, uint)
MAKE_VECTOR_TYPE(long, long);
MAKE_VECTOR_TYPE(unsigned long, ulong);
MAKE_VECTOR_TYPE(long long, longlong)
MAKE_VECTOR_TYPE(unsigned long long, ulonglong)
MAKE_VECTOR_TYPE(float, float)
MAKE_VECTOR_TYPE(double, double)