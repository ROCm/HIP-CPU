/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include <cstdint>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT DIM3
        struct Dim3 final {
            // DATA
            std::uint32_t x;
            std::uint32_t y;
            std::uint32_t z;

            // FRIENDS - COMPUTATIONAL BASIS
            friend
            inline
            constexpr
            bool operator<(const Dim3& x, const Dim3& y) noexcept
            {
                if (x.x < y.x) return true;
                if (y.x < x.x) return false;
                if (x.y < y.y) return true;
                if (y.y < x.x) return false;
                if (y.z < x.z) return false;
                return true;
            }
            friend
            inline
            constexpr
            bool operator==(const Dim3& x, const Dim3& y) noexcept
            {
                return x.x == y.x && x.y == y.y && x.z == y.z;
            }

            // FRIENDS - ACCESSORS
            friend
            inline
            constexpr
            std::uint32_t count(const Dim3& x) noexcept
            {   // TODO: add overflow checking.
                return x.x * x.y * x.z;
            }
            friend
            inline
            constexpr
            Dim3 extrude(const Dim3& x, std::uint32_t flat_index) noexcept
            {
                if (flat_index == 0u) return {0, 0, 0};
                if (x.y == 1 && x.z == 1) return {flat_index, 0, 0};

                const auto a{flat_index / (x.y * x.z)};
                const auto b{(flat_index - a * x.y * x.z) / x.z};
                const auto c{flat_index - a * x.y * x.z - b * x.z};

                return {a, b, c};
            }

            // CREATORS
            constexpr
            Dim3() noexcept : x{1}, y{1}, z{1} {};
            constexpr
            Dim3(
                std::uint32_t xx,
                std::uint32_t yy = 1u,
                std::uint32_t zz = 1u) noexcept
                : x{xx}, y{yy}, z{zz}
            {}
            Dim3(const Dim3&) = default;
            Dim3(Dim3&&) = default;
            ~Dim3() = default;

            // MANIPULATORS
            Dim3& operator=(const Dim3&) = default;
            Dim3& operator=(Dim3&&) = default;
        };
        // END STRUCT DIM3
    } // Namespace hip::detail.
} // Namespace hip.