/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#if !defined(NOMINMAX)
    #define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
    #define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT SYSTEM
        // STATICS
        inline
        typename System::Architecture System::architecture() noexcept
        {
            SYSTEM_INFO tmp{};
            GetNativeSystemInfo(&tmp);

            switch (tmp.wProcessorArchitecture) {
            case PROCESSOR_ARCHITECTURE_INTEL: return Architecture::x86;
            case PROCESSOR_ARCHITECTURE_ARM: return Architecture::arm;
            case PROCESSOR_ARCHITECTURE_AMD64: return Architecture::x64;
            case PROCESSOR_ARCHITECTURE_ARM64: return Architecture::arm64;
            default: return Architecture::generic;
            }
        }

        inline
        std::uint32_t System::core_count() noexcept
        {
            DWORD byte_cnt{};
            GetLogicalProcessorInformationEx(
                RelationProcessorCore, nullptr, &byte_cnt);

            std::uint32_t r{};

            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return r;

            using I = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            auto tmp{std::make_unique<std::byte[]>(byte_cnt)};
            if (!GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<I*>(tmp.get()),
                &byte_cnt)) {
                return r;
            }

            auto it{tmp.get()};
            while (byte_cnt) {
                auto x{reinterpret_cast<I*>(it)};

                ++r;

                it += x->Size;
                byte_cnt -= x->Size;
            }

            return r;
        }

        inline
        decltype(auto) System::cpu_cache() noexcept
        {
            struct {
                struct { DWORD size; } instruction;
                struct { DWORD size; } l1;
                struct { DWORD size; } l2;
                struct { DWORD size; } l3;
            } r{};

            DWORD byte_cnt{};
            GetLogicalProcessorInformationEx(RelationCache, nullptr, &byte_cnt);

            if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) return r;

            using I = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            auto tmp{std::make_unique<std::byte[]>(byte_cnt)};
            if (!GetLogicalProcessorInformationEx(
                RelationCache, reinterpret_cast<I*>(tmp.get()), &byte_cnt)) {
                return r;
            }

            auto it{tmp.get()};
            while (byte_cnt) {
                auto x{reinterpret_cast<I*>(it)};

                if (x->Cache.Type == CacheInstruction) {
                    r.instruction.size = x->Cache.CacheSize;
                }
                else {
                    switch (x->Cache.Level) {
                    case 1: r.l1.size = x->Cache.CacheSize; break;
                    case 2: r.l2.size = x->Cache.CacheSize; break;
                    case 3: r.l3.size = x->Cache.CacheSize; break;
                    default: break;
                    }
                }

                it += x->Size;
                byte_cnt -= x->Size;
            }

            return r;
        }

        inline
        decltype(auto) System::cpu_frequency() noexcept
        {
            static constexpr auto key{
                R"(HARDWARE\DESCRIPTION\System\CentralProcessor\0)"};
            static constexpr auto sub_key{"~MHz"};

            DWORD sz{};
            DWORD r{};

            if (RegGetValue(
                HKEY_LOCAL_MACHINE,
                key,
                sub_key,
                RRF_RT_DWORD,
                nullptr,
                nullptr,
                &sz) != ERROR_SUCCESS) {
                return r;
            }

            RegGetValue(
                HKEY_LOCAL_MACHINE,
                key,
                sub_key,
                RRF_RT_DWORD,
                nullptr,
                &r,
                &sz);

            return r;
        }

        inline
        std::string System::cpu_name() noexcept
        {
            static constexpr auto key{
                R"(HARDWARE\DESCRIPTION\System\CentralProcessor\0)"};
            static constexpr auto sub_key{"ProcessorNameString"};

            DWORD sz{};

            if (RegGetValue(
                HKEY_LOCAL_MACHINE,
                key,
                sub_key,
                RRF_RT_REG_SZ,
                nullptr,
                nullptr,
                &sz) != ERROR_SUCCESS) {
                return {};
            }

            std::string r(sz, '\0');

            if (RegGetValue(
                HKEY_LOCAL_MACHINE,
                key,
                sub_key,
                RRF_RT_REG_SZ,
                nullptr,
                std::data(r),
                &sz) != ERROR_SUCCESS) {
                return {};
            }

            r.resize(sz);

            return r;
        }

        inline
        decltype(auto) System::memory() noexcept
        {
            MEMORYSTATUSEX tmp{sizeof(MEMORYSTATUSEX)};

            struct { DWORDLONG available; DWORDLONG total; } r{};

            if (!GlobalMemoryStatusEx(&tmp)) return r;

            r.available = tmp.ullAvailPhys;
            r.total = tmp.ullTotalPhys;

            return r;
        }

        inline
        std::uint8_t System::simd_width() noexcept
        {   // TODO: should this actually look at the underlying HW?
            //       ARM et al.?
            #if defined(__AVX512F__)
                return 16;
            #elif defined(__AVX__) || defined(__AVX2__)
                return 8;
            #else
                return 4;
            #endif
        }
        // END STRUCT SYSTEM
    } // Namespace hip::detail.
} // Namespace hip.