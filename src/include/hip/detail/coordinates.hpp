/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "grid_launch.hpp"
#include "tile.hpp"
#include "types.hpp"

#include <cstdint>

namespace hip
{
    namespace detail
    {
        // BEGIN CLASS COORDINATES
        template<Dim3 (*fn)() noexcept>
        class Coordinates final {
            // IMPLEMENTATION - NESTED TYPES
            struct X final {
                __forceinline__
                constexpr
                operator std::uint32_t() const noexcept { return fn().x; }
            };
            struct Y final {
                __forceinline__
                constexpr
                operator std::uint32_t() const noexcept { return fn().y; }
            };
            struct Z final {
                __forceinline__
                constexpr
                operator std::uint32_t() const noexcept { return fn().z; }
            };
        public:
            inline static constexpr X x{};
            inline static constexpr Y y{};
            inline static constexpr Z z{};
        };
        // END CLASS COORDINATES

        struct BDim final {
            __forceinline__
            static
            Dim3 call() noexcept
            {
                return dimensions(Tile::this_tile());
            }
        };
        struct BIdx final {
            __forceinline__
            static
            Dim3 call() noexcept
            {
                return index(Tile::this_tile());
            }
        };
        struct GDim final {
            __forceinline__
            static
            Dim3 call() noexcept
            {
                return tile_count(domain(Tile::this_tile()));
            }
        };
        struct TIdx final {
            __forceinline__
            static
            Dim3 call() noexcept
            {   // TODO: redo.
                return extrude(
                    dimensions(Tile::this_tile()), id(Fiber::this_fiber()));
            }
        };

        using Block_dim = Coordinates<&BDim::call>;
        using Block_idx = Coordinates<&BIdx::call>;
        using Grid_dim = Coordinates<&GDim::call>;
        using Thread_idx = Coordinates<&TIdx::call>;
    } // Namespace hip::detail.
} // Namespace hip.