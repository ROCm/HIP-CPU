/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include <sys/utsname.h>
#include <unistd.h>

#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <string>
#include <string_view>
#include <vector>

namespace hip
{
    namespace detail
    {   // TODO: cache reads from /proc/*
        // BEGIN STRUCT SYSTEM
        // STATICS
        inline
        typename System::Architecture System::architecture() noexcept
        {   // TODO: revisit with more / more accurate strings.
            utsname tmp{};
            if (uname(&tmp)) return Architecture::generic;

            const std::string_view a{tmp.machine};

            if (a == "i686") return Architecture::x86;
            if (a == "x86_64") return Architecture::x64;
            if (a == "arm") return Architecture::arm;
            if (a == "aarch64") return Architecture::arm64;

            return Architecture::generic;
        }

        inline
        std::uint32_t System::core_count() noexcept
        {   // TODO: potentially brittle, re-assess.
            std::ifstream info{"/proc/cpuinfo"};

            if (!info) return {};

            std::string tmp{};
            while (std::getline(info, tmp)) {
                auto dx{tmp.find("cpu cores")};

                if (dx == std::string::npos) continue;

                while (!std::isdigit(tmp[++dx]));

                std::uint32_t r{};
                std::from_chars(
                    std::data(tmp) + dx, std::data(tmp) + std::size(tmp), r);

                return r;
            }

            return {};
        }

        inline
        decltype(auto) System::cpu_cache() noexcept
        {
            struct {
                struct { std::int32_t size; } instruction;
                struct { std::int32_t size; } l1;
                struct { std::int32_t size; } l2;
                struct { std::int32_t size; } l3;
            } r{};

            r.instruction.size = sysconf(_SC_LEVEL1_ICACHE_SIZE);
            r.l1.size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
            r.l2.size = sysconf(_SC_LEVEL2_CACHE_SIZE);
            r.l3.size = sysconf(_SC_LEVEL3_CACHE_SIZE);

            if (r.instruction.size == -1) r = {};
            else if (r.l1.size == -1) r = {};
            else if (r.l2.size == -1) r = {};
            else if (r.l3.size == -1) r = {};

            return r;
        }

        inline
        decltype(auto) System::cpu_frequency() noexcept
        {   // TODO: potentially brittle, re-assess.
            std::ifstream info{"/proc/cpuinfo"};

            if (!info) return std::uint32_t{};

            std::string tmp{};
            while (std::getline(info, tmp)) {
                auto dx{tmp.find("cpu MHz")};

                if (dx == std::string::npos) continue;

                dx += std::size("cpu MHz");

                while (!std::isdigit(tmp[++dx]));

                std::uint32_t r{};
                std::from_chars(
                    std::data(tmp) + dx, std::data(tmp) + std::size(tmp), r);

                return r;
            }

            return std::uint32_t{};
        }

        inline
        std::string System::cpu_name() noexcept
        {   // TODO: potentially brittle, re-assess.
            std::ifstream info{"/proc/cpuinfo"};

            if (!info) return {};

            std::string tmp{};
            while (std::getline(info, tmp)) {
                auto dx{tmp.find("model name")};

                if (dx == std::string::npos) continue;

                dx += std::size("model name");

                while (!std::isalpha(tmp[++dx]));

                return tmp.substr(dx);
            }

            return {};
        }

        inline
        decltype(auto) System::memory() noexcept
        {
            struct { std::int64_t available; std::int64_t total; } r{
                sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGESIZE),
                sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)
            };

            if (r.available < 0 || r.total < 0) r = {};

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