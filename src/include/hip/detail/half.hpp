/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "../../../../include/hip/hip_defines.h"
#include "helpers.hpp"
#include "vector.hpp"

#if defined(_MSC_VER)
    #pragma warning(push, 0)
#else
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "../../../../external/half/half.hpp"
#if defined(_MSC_VER)
    #pragma warning(pop)
#else
    #pragma GCC diagnostic pop
#endif

#include <type_traits>

namespace std
{
    template<> struct is_floating_point<half_float::half> : std::true_type {};
    template<> struct is_signed<half_float::half> : std::true_type {};
} // Namespace std.

namespace hip
{
    namespace detail
    {
        using half = half_float::half;
        using half2 = Vector_type<half, 2>;
        using half_float::half_cast;

        inline
        half cos(half x) noexcept
        {
            return half_float::cos(x);
        }

        inline
        half exp(half x) noexcept
        {
            return half_float::exp(x);
        }

        inline
        half2 exp(half2 x) noexcept
        {
            return {half_float::exp(x.x), half_float::exp(x.y)};
        }

        inline
        half log(half x) noexcept
        {
            return half_float::log(x);
        }

        inline
        half sin(half x) noexcept
        {
            return half_float::sin(x);
        }

        inline
        half sqrt(half x) noexcept
        {
            return half_float::sqrt(x);
        }
    } // Namespace hip::detail.
} // Namespace hip.