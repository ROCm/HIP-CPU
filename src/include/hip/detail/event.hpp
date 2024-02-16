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
#include <type_traits>
#include <utility>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT EVENT
        class Event final {
            // DATA
            decltype(std::declval<Task>().get_future()) is_done_{};
            decltype(std::chrono::high_resolution_clock::now()) time_{};
            bool all_synch_{false};

            // FRIENDS - MANIPULATORS
            template<
                typename T,
                std::enable_if_t<std::is_constructible_v<
                    decltype(is_done_),
                    decltype(std::move(std::declval<T>()))>>* = nullptr>
            friend
            inline
            void add_done_signal(Event& x, T awaitable_signal)
            {
                return x.add_done_signal(std::move(awaitable_signal));
            }
            friend
            inline
            void mark_as_all_synchronising(Event& x) noexcept
            {
                return x.mark_as_all_synchronising();
            }
            friend
            inline
            void update_timestamp(Event& x) noexcept
            {
                return x.update_timestamp();
            }

            // FRIENDS - ACCESSORS
            friend
            inline
            bool is_all_synchronising(const Event& x) noexcept
            {
                return x.is_all_synchronising();
            }
            friend
            inline
            decltype(auto) is_done(const Event& x) noexcept
            {
                return x.is_done();
            }
            friend
            inline
            decltype(auto) is_ready(const Event& x) noexcept
            {
                return x.is_ready();
            }
            friend
            inline
            decltype(auto) timestamp(const Event& x) noexcept
            {
                return x.timestamp();
            }
        public:
            // NESTED TYPES
            using awaitable_type = decltype(is_done_);
            using time_type = decltype(time_);

            // CREATORS
            Event() = default;
            explicit
            Event(hipEventFlags flags) noexcept;
            Event(const Event&) = delete;
            Event(Event&&) = default;
            ~Event() = default;

            // MANIPULATORS
            Event& operator=(const Event&) = delete;
            Event& operator=(Event&&) = default;
            void add_done_signal(awaitable_type signal);
            void mark_as_all_synchronising() noexcept;
            void update_timestamp() noexcept;

            // ACCESSORS
            bool is_all_synchronising() const noexcept;
            const awaitable_type& is_done() const noexcept;
            bool is_ready() const noexcept;
            time_type timestamp() const noexcept;
        };

        // CREATORS
        inline
        Event::Event(hipEventFlags /*flags*/) noexcept
            : is_done_{}, time_{}, all_synch_{false}
        {}

        // MANIPULATORS
        inline
        void Event::add_done_signal(awaitable_type x)
        {
            is_done_ = std::move(x);
        }

        inline
        void Event::mark_as_all_synchronising() noexcept
        {
            all_synch_ = true;
        }

        inline
        void Event::update_timestamp() noexcept
        {
            time_ = std::chrono::high_resolution_clock::now();
        }

        // ACCESSORS
        inline
        bool Event::is_all_synchronising() const noexcept
        {
            return all_synch_;
        }

        inline
        const typename Event::awaitable_type& Event::is_done() const noexcept
        {
            return is_done_;
        }

        inline
        bool Event::is_ready() const noexcept
        {
            // a default-constructed event does not correspond to any work,
            // so it is considered "ready"
            if (not is_done_.valid())
                return true;

            // check if the event is "ready"
            return is_done_.wait_for(std::chrono::seconds(0))
                == std::future_status::ready;
        }

        inline
        Event::time_type Event::timestamp() const noexcept
        {
            return time_;
        }
        // END STRUCT EVENT
    } // Namespace hip::detail.
} // Namespace hip.
