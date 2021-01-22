/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "fiber.hpp"
#include "tile.hpp"
#include "../../../../include/hip/hip_constants.h"

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

#include <atomic>
#include <bitset>
#include <chrono>
#include <climits>
#include <ctime>
#include <type_traits>

namespace hip
{
    namespace detail
    {
        inline
        std::uint64_t ballot(std::int32_t x) noexcept
        {
            const auto tidx{id(Fiber::this_fiber()) % warpSize};
            auto& lds{Tile::scratchpad<std::bitset<warpSize>, 1>()[0]};

            lds[tidx] = static_cast<bool>(x);

            barrier(Tile::this_tile());

            return lds.to_ullong();
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) <= sizeof(std::uint64_t)>* = nullptr>
        inline
        std::uint32_t bit_scan_forward(T x) noexcept
        {   // TODO: the standard library probably forwards to the intrinsics.
            [[maybe_unused]]
            constexpr auto bscan{[](T x) constexpr noexcept { // TODO: endianness.
                std::bitset<sizeof(T) * CHAR_BIT> tmp(x);
                for (decltype(std::size(tmp)) i = 0; i != std::size(tmp); ++i) {
                    if (tmp[i]) return static_cast<std::uint32_t>(i + 1);
                }

                return 0u;
            }};

            if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
                #if defined(_MSC_VER) && !defined(__clang__)
                    unsigned long r{0};
                    return static_cast<std::uint32_t>(
                        _BitScanForward(&r, x) ? (r + 1) : 0);
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_ffs)
                        return __builtin_ffs(x);
                    #else
                        return bscan(x);
                    #endif
                #else
                    return bscan(x);
                #endif
            }
            else {
                #if defined(_MSC_VER) && !defined(__clang__)
                    unsigned long r{0};
                    return static_cast<std::uint32_t>(
                        _BitScanForward64(&r, x) ? (r + 1) : 0);
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_ffs)
                        return __builtin_ffsll(x);
                    #else
                        return bscan(x);
                    #endif
                #else
                    return bscan(x);
                #endif
            }
        }

        inline
        decltype(auto) clock() noexcept
        {
            return std::clock();
        }

        inline
        decltype(auto) clock64() noexcept
        {
            return std::chrono::high_resolution_clock::
                now().time_since_epoch().count();
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        std::uint32_t count_leading_zeroes(T x) noexcept
        {
            if (x == T{0}) return sizeof(T) * CHAR_BIT;

            [[maybe_unused]]
            constexpr auto lzcnt{[](auto&& x) constexpr noexcept {
                std::bitset<sizeof(T) * CHAR_BIT> tmp(x);

                auto n{std::size(tmp)};
                while (n--) {
                    if (tmp[n]) {
                        return static_cast<std::uint32_t>(size(tmp) - n - 1);
                    }
                }

                return static_cast<std::uint32_t>(size(tmp));
            }};

            if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
                #if defined(_MSC_VER)
                    return __lzcnt(x);
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_clz)
                        return __builtin_clz(x);
                    #else
                        return lzcnt(x);
                    #endif
                #else
                    return lzcnt(x);
                #endif
            }
            else {
                #if defined(_MSC_VER)
                    return static_cast<std::uint32_t>(__lzcnt64(x));
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_clzll)
                        return __builtin_clzll(x);
                    #else
                        return lzcnt(x);
                    #endif
                #else
                    return lzcnt(x);
                #endif
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        std::uint32_t pop_count(T x) noexcept
        {
            [[maybe_unused]]
            constexpr auto popcnt{[](auto&& x) constexpr noexcept {
                return std::bitset<sizeof(T) * CHAR_BIT>(x).count();
            }};

            if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
                #if defined(_MSC_VER)
                    return __popcnt(x);
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_popcount)
                        return __builtin_popcount(x);
                    #else
                        return popcnt(x);
                    #endif
                #else
                    return popcnt(x);
                #endif
            }
            else {
                #if defined(_MSC_VER)
                    return static_cast<std::uint32_t>(__popcnt64(x));
                #elif defined(__has_builtin)
                    #if __has_builtin(__builtin_popcountll)
                        return __builtin_popcountll(x);
                    #else
                        return popcnt(x);
                    #endif
                #else
                    return popcnt(x);
                #endif
            }
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
                (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
        inline
        T shuffle(T x, std::int32_t src, std::int32_t w) noexcept
        {
            const auto tidx{id(Fiber::this_fiber()) % warpSize};

            Tile::scratchpad<T>()[tidx] = x;

            barrier(Tile::this_tile());

            const auto sidx{(tidx / w * w) + src};

            return (src < 0 || sidx >= w) ? x : Tile::scratchpad<T>()[sidx];
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
                (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
        inline
        T shuffle_down(T x, std::int32_t dx, std::int32_t w) noexcept
        {   // TODO: incorrect with large negative offsets, revisit.
            const auto tidx{id(Fiber::this_fiber())};
            Tile::scratchpad<T>()[tidx] = x;

            Tile::this_tile().barrier();

            const auto sidx{(tidx / w * w) + (tidx % w) + dx};

            return (sidx < 0 || sidx >= w) ? x : Tile::scratchpad<T>()[sidx];
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
                (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
        inline
        T shuffle_xor(T x, std::int32_t src, std::int32_t w) noexcept
        {   // TODO: probably incorrect, revisit.
            const auto tidx{id(Fiber::this_fiber())};
            Tile::scratchpad<T>()[tidx] = x;

            Tile::this_tile().barrier();

            return (src < 0) ? x : Tile::scratchpad<T>()[(tidx / w * w) ^ src];
        }

        inline
        void thread_fence() noexcept
        {   // TODO: tighten check.
            #if __cplusplus > 201703L
                return std::atomic_thread_fence(std::memory_order::seq_cst);
            #else
                return std::atomic_thread_fence(
                    std::memory_order::memory_order_seq_cst);
            #endif
        }

        inline
        void thread_fence_block() noexcept
        {
            return thread_fence();
        }

        inline
        void thread_fence_system() noexcept
        {
            return thread_fence();
        }
    } // Namespace hip::detail;
} // Namespace hip.