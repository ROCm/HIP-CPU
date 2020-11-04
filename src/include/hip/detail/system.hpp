/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT SYSTEM
        struct System final { // TODO: group CPU details in a struct.
            // NESTED TYPES
            enum class Architecture { x86, x64, arm, arm64, generic };

            // STATICS
            static
            Architecture architecture() noexcept;
            static
            std::uint32_t core_count() noexcept;
            static
            decltype(auto) cpu_cache() noexcept;
            static
            decltype(auto) cpu_frequency() noexcept; // MHz
            static
            std::string cpu_name() noexcept;
            static
            decltype(auto) memory() noexcept;
            static
            std::uint8_t simd_width() noexcept;
        };
        // END STRUCT SYSTEM
    } // Namespace hip::detail.
} // Namespace hip.

#if defined(_WIN32)
    #include "system_windows.inl"
#else
    #include "system_posix.inl"
#endif
