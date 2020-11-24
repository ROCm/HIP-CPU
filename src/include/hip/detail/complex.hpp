/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "vector.hpp"

#include <cmath>

namespace hip
{
    namespace detail
    {   // TODO: perhaps we can use std::complex here.
        template<typename T>
        using complex = Vector_type<T, 2>;

        template<typename T>
        inline
        T absolute_square(complex<T> x) noexcept
        {
            return x.x * x.x + x.y * x.y;
        }

        template<typename T>
        inline
        T abs(complex<T> x) noexcept
        {
            return std::sqrt(absolute_square(x));
        }

        template<typename T>
        inline
        complex<T> conjugate(complex<T> x) noexcept
        {
            x.y = -x.y;

            return x;
        }

        template<typename T>
        inline
        complex<T> divide(complex<T> x, complex<T> y) noexcept
        {
            const auto abs_sq{absolute_square(y)};

            return {(x.x * y.x + x.y * y.y) / abs_sq, (x.y * y.x - x.x * y.y)};
        }

        template<typename T>
        inline
        T imag(complex<T> x) noexcept
        {
            return x.y;
        }

        template<typename T>
        inline
        complex<T> multiply(complex<T> x, complex<T> y) noexcept
        {
            return {x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x};
        }

        template<typename T>
        inline
        T real(complex<T> x) noexcept
        {
            return x.x;
        }
    } // Namespace hip::detail.
} // Namespace hip.