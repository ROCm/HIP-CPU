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
            thread_local static std::bitset<64> r{};

            const auto tidx{id(Fiber::this_fiber()) % warpSize};

            r[tidx] = static_cast<bool>(x);

            barrier(Tile::this_tile());

            return r.to_ullong();
        }

        template<
            typename T,
            std::enable_if_t<
                std::is_integral_v<T> &&
                sizeof(T) <= sizeof(std::uint64_t)>* = nullptr>
        inline
        std::uint32_t bit_scan_forward(T x) noexcept
        {
            [[maybe_unused]]
            constexpr auto bscan{[](T x) constexpr noexcept -> std::uint32_t { // TODO: endianness.
                std::bitset<sizeof(T) / CHAR_BIT> tmp(x);
                for (auto i = std::size(tmp) - 1; i != 0; --i) {
                    if (tmp[i]) return i + 1;
                }

                return 0;
            }};

            #if defined(_MSC_VER) && !defined(__clang__)
                unsigned long r{0};

                if constexpr (sizeof(T) <= 4) _BitScanForward(&r, x);
                else _BitScanForward64(&r, x);

                return r;
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
        std::int32_t count_leading_zeroes(T x) noexcept
        {
            [[maybe_unused]]
            constexpr auto lz_cnt{[](auto&& x) constexpr noexcept { // TODO: endianness.
                std::bitset<sizeof(T) / CHAR_BIT> tmp(x);

                auto cnt{0};
                for (auto i = 0u; i != std::size(tmp) && !tmp[i]; ++i, ++cnt);

                return cnt;
            }};

            #if defined(_MSC_VER)
                if constexpr (sizeof(T) == sizeof(std::int32_t)) {
                    return __lzcnt(x);
                }
                else return __lzcnt64(x);
            #elif defined(__has_builtin)
                #if __has_builtin(__builtin_clz)
                    return __builtin_clz(x);
                #else
                    return lz_cnt(x);
                #endif
            #else
                return lz_cnt(x);
            #endif
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

            return Tile::scratchpad<T>()[(tidx / w * w) + src];
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
                (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
        inline
        T shuffle_down(T x, std::int32_t dx, std::int32_t w = warpSize) noexcept
        {
            const auto tidx{id(Fiber::this_fiber())};
            Tile::scratchpad<T>()[tidx] = x;

            Tile::this_tile().barrier();

            return Tile::scratchpad<T>()[(tidx / w * w) + (tidx % w) + dx];
        }

        template<
            typename T,
            std::enable_if_t<
                (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
                (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
        inline
        T shuffle_xor(T x, std::int32_t src, std::int32_t w = warpSize) noexcept
        {   // TODO: probably incorrect, revisit.
            const auto tidx{id(Fiber::this_fiber())};
            Tile::scratchpad<T>()[tidx] = x;

            Tile::this_tile().barrier();

            return Tile::scratchpad<T>()[(tidx / w * w) ^ src];
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