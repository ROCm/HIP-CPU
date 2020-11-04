/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#if defined(_MSC_VER) && !defined(_MATH_DEFINES_DEFINED)
    #define _MATH_DEFINES_DEFINED
#endif
#include <cmath>
#include <cstdint>
#include <execution>
#include <numeric>
#include <type_traits>
#include <version>
#if defined(__cpp_lib_math_constants)
    #include <numbers>

    namespace hip
    {
        namespace detail
        {
            using std::numbers::pi_v;
        } // Namespace hip::detail.
    } // Namespace hip.
#else
    namespace hip
    {
        namespace detail
        {
            #if defined(M_PI)
                template<typename T>
                inline constexpr T pi_v{M_PI};
            #else
                template<typename T>
                inline constexpr T pi_v{3.14159265358979323846};
            #endif
        } // Namespace hip::detail.
    } // Namespace hip.
#endif

namespace hip
{
    namespace detail
    {
        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T erfinv(T x) noexcept
        {   // From: https://stackoverflow.com/a/49743348/13159551, placeholder
            const auto t{std::log(std::fma(x, T{0} - x, T{1}))};
            T p;
            if (std::fabs(x) > T{6.125}) {
                p = T{3.03697567e-10};
                p = std::fma(p, t, T{2.93243101e-8});
                p = std::fma(p, t, T{1.22150334e-6});
                p = std::fma(p, t, T{2.84108955e-5});
                p = std::fma(p, t, T{3.93552968e-4});
                p = std::fma(p, t, T{3.02698812e-3});
                p = std::fma(p, t, T{4.83185798e-3});
                p = std::fma(p, t, T{-2.64646143e-1});
                p = std::fma(p, t, T{8.40016484e-1});
            }
            else {
                p = T{5.43877832e-9};
                p = std::fma(p, t, T{1.43286059e-7});
                p = std::fma(p, t, T{1.22775396e-6});
                p = std::fma(p, t, T{1.12962631e-7});
                p = std::fma(p, t, T{-5.61531961e-5});
                p = std::fma(p, t, T{-1.47697705e-4});
                p = std::fma(p, t, T{2.31468701e-3});
                p = std::fma(p, t, T{1.15392562e-2});
                p = std::fma(p, t, T{-2.32015476e-1});
                p = std::fma(p, t, T{8.86226892e-1});
            }

            return x * p;
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T erfcinv(T x) noexcept
        {
            return erfinv(T{1} - x);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T erfcx(T x) noexcept
        {
            return std::exp(x * x) * std::erfc(x);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T fdivide(T x, T y) noexcept
        {
            return x / y;
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T log2(T x) noexcept
        {
            return std::log2(x);
        }

        inline
        std::int32_t mul24(std::int32_t x, std::int32_t y) noexcept
        {
            struct int24_t { std::int32_t x : 24; };

            return int24_t{x}.x * int24_t{y}.x;
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T norm(std::int32_t dim, const T* p) noexcept
        {
            return std::sqrt(std::transform_reduce(
                /*std::execution::unseq,*/ p, p + dim, p, T{0}));
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T normcdf(T x) noexcept
        {
            return std::erfc(-x / std::sqrt(T{2})) / T{2};
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T normcdfinv(T x) noexcept
        {
            return -std::sqrt(T{2}) * erfcinv(T{2} * x);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T norm3d(T x, T y, T z) noexcept
        {
            return std::sqrt(x * x + y * y + z * z);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T norm4d(T x, T y, T z, T w) noexcept
        {
            return std::sqrt(x * x + y * y + z * z + w * w);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T pow(T x, T y) noexcept
        {
            return std::pow(x, y);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T rcbrt(T x) noexcept
        {
            return T{1} / std::cbrt(x);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T rhypot(T x, T y) noexcept
        {
            return T{1} / std::hypot(x, y);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T rnorm(std::int32_t dim, const T* p) noexcept
        {
            return T{1} / norm(dim, p);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T rnorm3d(T x, T y, T z) noexcept
        {
            return T{1} / norm3d(x, y, z);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T rnorm4d(T x, T y, T z, T w) noexcept
        {
            return T{1} / norm4d(x, y, z, w);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        void sincos(T x, T* psin, T* pcos) noexcept
        {
            if (!psin || !pcos) std::abort();

            *psin = std::sin(x);
            *pcos = std::cos(x);
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        void sincospi(T x, T* psin, T* pcos) noexcept
        {
            if (!psin || !pcos) std::abort();

            *psin = std::sin(pi_v<T> * x);
            *pcos = std::cos(pi_v<T> * x);
        }
    } // Namespace hip::detail.
} // Namespace hip.