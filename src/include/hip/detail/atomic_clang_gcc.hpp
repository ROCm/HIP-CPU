/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "helpers.hpp"

#include <cstdint>
#include <type_traits>

namespace hip
{
    namespace detail
    {
        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_add(T* p, T v) noexcept
        {
            return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_cas(T* p, T cmp, T v) noexcept
        {
            __atomic_compare_exchange(
                p, &cmp, &v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

            return cmp;
        }

        template<
            typename T,
            std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
        inline
        T atomic_cas_add(T* p, T v) noexcept
        {
            using UI = std::conditional_t<
                sizeof(T) == sizeof(std::uint64_t),
                std::uint64_t,
                std::uint32_t>;

            auto uaddr{reinterpret_cast<UI*>(p)};
            auto r{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};

            UI old;
            do {
                old = __atomic_load_n(uaddr, __ATOMIC_RELAXED);

                if (r != old) { r = old; continue; }

                const auto tmp{v + bit_cast<T>(r)};

                r = atomic_cas(uaddr, r, bit_cast<UI>(tmp));

                if (r == old) break;
            } while (true);

            return bit_cast<T>(r);
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) >= sizeof(std::uint32_t)>* = nullptr>
        inline
        T atomic_and(T* p, T v) noexcept
        {
            return __atomic_fetch_and(p, v, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) >= sizeof(std::uint32_t)>* = nullptr>
        inline
        T atomic_dec(T* p) noexcept
        {   // TODO: incorrect saturation / wrapping behaviour.
            return __atomic_fetch_sub(p, T{1}, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> && sizeof(T) >= sizeof(uint32_t)) ||
                std::is_same_v<T, float>>* = nullptr>
        inline
        T atomic_exchange(T* p, T v) noexcept
        {
            T r;
            __atomic_exchange(p, &v, &r, __ATOMIC_RELAXED);

            return r;
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_inc(T* p) noexcept
        {   // TODO: wrapping behaviour unimplemented.
            return atomic_add(p, T{1});
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_max(T* p, T v) noexcept
        {
            #if defined (__has_builtin)
                #if __has_builtin(__atomic_fetch_max)
                    return __atomic_fetch_max(p, v, __ATOMIC_RELAXED);
                #elif __has_builtin(__sync_fetch_and_max)
                    if constexpr (std::is_unsigned_v<T>) {
                        return __sync_fetch_and_umax(p, v);
                    }
                    else return __sync_fetch_and_max(p, v);
                #endif
            #endif

            auto tmp{atomic_add(p, T{0})};
            while (tmp < v) {
                const auto tmp1{atomic_add(p, T{0})};

                if (tmp1 != tmp) { tmp = tmp1; continue; }

                tmp = atomic_cas(p, tmp, v);
            }

            return tmp;
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_min(T* p, T v) noexcept
        {
            #if defined (__has_builtin)
                #if __has_builtin(__atomic_fetch_min)
                    return __atomic_fetch_min(p, v, __ATOMIC_RELAXED);
                #elif __has_builtin(__sync_fetch_and_min)
                    if constexpr (std::is_unsigned_v<T>) {
                        return __sync_fetch_and_umin(p, v);
                    }
                    else return __sync_fetch_and_min(p, v);
                #endif
            #endif

            auto tmp{atomic_add(p, T{0})};
            while (v < tmp) {
                const auto tmp1{atomic_add(p, T{0})};

                if (tmp1 != tmp) { tmp = tmp1; continue; }

                tmp = atomic_cas(p, tmp, v);
            }

            return tmp;
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) >= sizeof(std::uint32_t)>* = nullptr>
        inline
        T atomic_or(T* p, T v) noexcept
        {
            return __atomic_fetch_or(p, v, __ATOMIC_RELAXED);
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_sub(T* p, T v) noexcept
        {
            return __atomic_fetch_sub(p, v, __ATOMIC_RELAXED);
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_xor(T* p, T v) noexcept
        {
            return __atomic_fetch_xor(p, v, __ATOMIC_RELAXED);
        }
    } // Namespace hip::detail.
} // Namespace hip.