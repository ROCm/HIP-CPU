/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "task.hpp"
#include "../../../../include/hip/hip_enums.h"

#include <chrono>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT EVENT
        class Event final {
            // DATA
            decltype(std::declval<Task>().get_future()) done{};
            decltype(std::chrono::high_resolution_clock::now()) time{};
            /*bool has_timing_{false};*/

            // FRIENDS - ACCESSORS
            friend
            inline
            decltype(auto) is_done(const Event& x) noexcept { return (x.done); }
            friend
            inline
            decltype(auto) time(const Event& x) noexcept { return (x.time); }

            // FRIENDS - MANIPULATORS
            friend
            inline
            decltype(auto) is_done(Event& x) noexcept { return (x.done); }
            friend
            inline
            decltype(auto) time(Event& x) noexcept { return (x.time); }
        public:
            // CREATORS
            Event() = default;
            explicit
            Event(hipEventFlags flags) noexcept;
            Event(const Event&) = delete;
            Event(Event&&) = default;
            ~Event() = default;
        };

        // CREATORS
        inline
        Event::Event(hipEventFlags /*flags*/) noexcept
            : done{}, time{}/*, has_timing_{flags != hipEventDisableTiming}*/
        {}
        // END STRUCT EVENT
    } // Namespace hip::detail.
} // Namespace hip.