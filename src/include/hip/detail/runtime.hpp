/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "event.hpp"
#include "stream.hpp"
#include "task.hpp"
#include "../../../../include/hip/hip_defines.h"
#include "../../../../include/hip/hip_enums.h"

#include <algorithm>
#include <execution>
#include <forward_list>
#include <future>
#include <iterator>
#include <limits>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable:4251) // TODO: temporary.
#endif

namespace hip
{
    namespace detail
    {
        // BEGIN CLASS RUNTIME
        class __HIP_API__ Runtime final {
            // NESTED TYPES
            struct Cleaner_ {
                // CREATORS
                Cleaner_() noexcept;
                ~Cleaner_();
            };

            // DATA - STATICS
            inline static constexpr auto max_n_{
                std::numeric_limits<Stream::ssize_t>::max()};
            // inline static thread_local hipError_t last_error_{};
            inline static Stream internal_stream_{};
            inline static std::forward_list<Stream> streams_{};
            inline static const Cleaner_ cleaner{};

            // IMPLEMENTATION - STATICS
            static
            hipError_t& last_error_() noexcept;
            static
            void pause_or_yield_() noexcept;
            static
            std::thread& processor_();
            static
            void wait_all_streams_();
        public:
            // STATICS
            static
            std::future<void> destroy_stream_async(Stream* s);
            static
            hipError_t last_error() noexcept;
            static
            std::future<Stream*> make_stream_async();
            static
            Stream* null_stream();
            static
            void push_task(Event* p, Stream* s);
            static
            hipError_t set_last_error(hipError_t e) noexcept;
            static
            void synchronize();
        };

        // NESTED TYPES
        // BEGIN CLASS RUNTIME::CLEANER_
        inline
        Runtime::Cleaner_::Cleaner_() noexcept
        {
            last_error_();
        }

        inline
        Runtime::Cleaner_::~Cleaner_()
        {
            Task poison{[](auto&& p) { p = true; }};
            auto fut{poison.get_future()};

            internal_stream_.enqueue(std::move(poison));

            if (processor_().joinable()) processor_().join();

            fut.get();
        }
        // END CLASS RUNTIME::CLEANER_

        // IMPLEMENTATION - STATICS
        inline
        hipError_t& Runtime::last_error_() noexcept
        {
            static thread_local hipError_t r{hipSuccess};

            return r;
        }

        inline
        void Runtime::pause_or_yield_() noexcept
        {
            #if defined(YieldProcessor)
                return YieldProcessor();
            #elif defined(__has_builtin)
                #if __has_builtin(__builtin_ia32_pause)
                    return __builtin_ia32_pause();
                #else
                    return std::this_thread::yield();
                #endif
            #else
                return std::this_thread::yield();
            #endif
        }

        inline
        std::thread& Runtime::processor_()
        {
            static std::thread r{[]() {
                std::vector<Task> t;
                do {
                    t.clear();

                    internal_stream_.try_dequeue_bulk(
                        std::back_inserter(t), max_n_);
                    for (auto&& x : t) {
                        bool poison{};
                        x(poison);

                        if (poison) return;
                    }

                    // wait_all_streams_();

                    const auto backoff{
                        std::empty(t) &&
                        std::none_of(
                            std::cbegin(streams_),
                            std::cend(streams_),
                            [](auto&& x) { return x.size_approx(); })};

                    if (!backoff) wait_all_streams_();
                    else {
                        static std::minstd_rand g{std::random_device{}()};
                        static std::uniform_int_distribution<std::uint32_t> d{
                            3, 1031};

                        for (auto i = 0u, n = d(g); i != n; ++i) {
                            pause_or_yield_();
                        }
                    }
                } while (true);
            }};

            return r;
        }

        inline
        void Runtime::wait_all_streams_()
        {
            if (processor_().joinable()) processor_().detach();

            std::for_each(
                std::execution::par,
                std::begin(streams_),
                std::end(streams_),
                [](auto&& x) {
                static thread_local std::vector<Task> t;

                x.try_dequeue_bulk(std::back_inserter(t), max_n_);

                for (auto&& y : t) { bool nop{}; y(nop); }

                t.clear();
            });
        }

        // STATICS
        inline
        std::future<void> Runtime::destroy_stream_async(Stream* s)
        {
            Task r{[=](auto&&) {
                streams_.remove_if([=](auto&& x) { return &x == s; });
            }};
            auto fut{r.get_future()};

            internal_stream_.enqueue(std::move(r));

            return fut;
        }

        inline
        hipError_t Runtime::last_error() noexcept
        {
            return last_error_();
        }

        inline
        std::future<Stream*> Runtime::make_stream_async()
        {   // TODO: use smart pointer.
            auto p{new std::promise<Stream*>};
            auto fut{p->get_future()};

            internal_stream_.enqueue(Task{[=](auto&&) mutable {
                p->set_value(&streams_.emplace_front());
                delete p;
            }});

            if (processor_().joinable()) processor_().detach();

            return fut;
        }

        inline
        Stream* Runtime::null_stream()
        {
            static auto& r{streams_.emplace_front()};

            if (processor_().joinable()) processor_().detach();

            return &r;
        }

        inline
        void Runtime::push_task(Event* p, Stream* s)
        {
            if (!s) {
                s = null_stream();
                mark_as_all_synchronising(*p);
            }

            Task r{[=](auto&&) { update_timestamp(*p); }};
            add_done_signal(*p, r.get_future());

            s->enqueue(std::move(r));
        }

        inline
        hipError_t Runtime::set_last_error(hipError_t e) noexcept
        {
            return std::exchange(last_error_(), e);
        }

        inline
        void Runtime::synchronize()
        {   // TODO: redo, this induces ordering requirements on the processor.
            Task r{[](auto&&) { wait_all_streams_(); }};
            auto fut{r.get_future()};

            internal_stream_.enqueue(std::move(r));

            if (processor_().joinable()) processor_().detach();

            return fut.get();
        }
        // END CLASS RUNTIME
    } // Namespace hip::detail.
} // Namespace hip.

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif