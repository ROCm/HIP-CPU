/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "helpers.hpp"

#include <intrin.h>

#include <cstdint>
#include <type_traits>

// TODO: use ARM support for _nf.

namespace hip
{
    namespace detail
    {
        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_add(T* p, T v) noexcept
        {
            using U = std::conditional_t<
                std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
                    std::is_same_v<T, long long>,
                T,
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>>;

            if constexpr (std::is_same_v<T, U>) {
                return _InterlockedExchangeAdd(p, v);
            }
            else if constexpr (sizeof(U) == sizeof(long)) {
                return bit_cast<T>(_InterlockedExchangeAdd(
                    reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
            else {
                return bit_cast<T>(_InterlockedExchangeAdd64(
                    reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_cas(T* p, T cmp, T v) noexcept
        {
            using U = std::conditional_t<
                std::is_same_v<T, long long> ||
                    std::is_same_v<T, unsigned long> || std::is_same_v<T, long>,
                T,
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>>;

            if constexpr (std::is_same_v<T, U>) {
                return _InterlockedCompareExchange(p, v, cmp);
            }
            else if constexpr (sizeof(T) == sizeof(long)) {
                return bit_cast<T>(_InterlockedCompareExchange(
                    reinterpret_cast<U*>(p),
                    bit_cast<U>(v),
                    bit_cast<U>(cmp)));
            }
            else if constexpr (sizeof(T) == sizeof(long long)) {
                return bit_cast<T>(_InterlockedCompareExchange64(
                    reinterpret_cast<U*>(p), bit_cast<U>(v), bit_cast<U>(cmp)));
            }
            else if constexpr (std::is_unsigned_v<T>) {
                return bit_cast<T>(_InterlockedCompareExchange16(
                    reinterpret_cast<short*>(p),
                    bit_cast<short>(v),
                    bit_cast<short>(cmp)));
            }
            else return _InterlockedCompareExchange16(p, v, cmp);
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
            auto r{atomic_add(uaddr, UI{0})};

            UI old;
            do {
                old = atomic_add(uaddr, UI{0});

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
            using U =
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>;

            if constexpr (sizeof(T) == sizeof(long)) {
                return bit_cast<T>(
                    _InterlockedAnd(reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
            else {
                return bit_cast<T>(
                    _InterlockedAnd64(reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) >= sizeof(std::uint32_t)>* = nullptr>
        inline
        T atomic_dec(T* p) noexcept
        {   // TODO: incorrect saturation / wrapping behaviour.
            using U =
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>;

            if constexpr (sizeof(T) == sizeof(long)) {
                return
                    bit_cast<T>(_InterlockedDecrement(reinterpret_cast<U*>(p)));
            }
            else {
                return bit_cast<T>(
                    _InterlockedDecrement64(reinterpret_cast<U*>(p)));
            }
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> && sizeof(T) >= sizeof(uint32_t)) ||
                std::is_same_v<T, float>>* = nullptr>
        inline
        T atomic_exchange(T* p, T v) noexcept
        {
            using U = std::conditional_t<
                std::is_same_v<T, long> ||
                    std::is_same_v<T, unsigned long> ||
                    std::is_same_v<T, long long>,
                T,
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>>;

            if constexpr (std::is_same_v<T, U>) {
                return _InterlockedExchange(p, v);
            }
            else if constexpr (sizeof(T) == sizeof(long)) {
                return bit_cast<T>(_InterlockedExchange(
                    reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
            else {
                return bit_cast<T>(_InterlockedExchange64(
                    reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_inc(T* p) noexcept
        {   // TODO: wrapping behaviour unimplemented.
            using U =
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>;

            if constexpr (sizeof(T) == sizeof(long)) {
                return
                    bit_cast<T>(_InterlockedIncrement(reinterpret_cast<U*>(p)));
            }
            else {
                return bit_cast<T>(
                    _InterlockedIncrement64(reinterpret_cast<U*>(p)));
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_max(T* p, T v) noexcept
        {
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
            using U =
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>;

            if constexpr (sizeof(U) == sizeof(long)) {
                return bit_cast<T>(_InterlockedOr(
                    reinterpret_cast<U*>(p), reinterpret_cast<U const&>(v)));
            }
            else {
                return bit_cast<T>(
                    _InterlockedOr64(reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_sub(T* p, T v) noexcept
        {
            if constexpr (std::is_unsigned_v<T>) {
                return atomic_add(
                    reinterpret_cast<std::make_signed_t<T>*>(p),
                    -bit_cast<std::make_signed_t<T>>(v));
            }
            else return atomic_add(p, -v);
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        T atomic_xor(T* p, T v) noexcept
        {
            using U =
                std::conditional_t<sizeof(T) == sizeof(long), long, long long>;

            if constexpr (sizeof(U) == sizeof(long)) {
                return bit_cast<T>(
                    _InterlockedXor(reinterpret_cast<U*>(p), bit_cast<T>(v)));
            }
            else {
                return bit_cast<T>(_InterlockedXor64(
                    reinterpret_cast<U*>(p), bit_cast<U>(v)));
            }
        }
    } // Namespace hip::detail.
} // Namespace hip.