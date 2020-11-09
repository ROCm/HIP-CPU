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
#include "../../../../include/hip/hip_enums.h"

#include <algorithm>
#include <execution>
#include <forward_list>
#include <future>
#include <iterator>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace hip
{
    namespace detail
    {
        // BEGIN CLASS RUNTIME
        class Runtime final {
            // NESTED TYPES
            struct Cleaner_ {
                // CREATORS
                Cleaner_() noexcept;
                ~Cleaner_();
            };

            // IMPLEMENTATION - STATICS
            static
            void process_tasks_();

            // DATA - STATICS
            inline static constexpr auto max_n_{
                std::numeric_limits<Stream::ssize_t>::max()};
            inline static thread_local hipError_t last_error_{};
            inline static Stream internal_stream_{};
            inline static std::forward_list<Stream> streams_{};
            inline static std::thread processor_{process_tasks_};
            inline static const Cleaner_ cleaner{};

            // IMPLEMENTATION - STATICS
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
            last_error_ = hipSuccess;
        }

        inline
        Runtime::Cleaner_::~Cleaner_()
        {
            internal_stream_.enqueue(Task{[](auto&& p) { p = true; }});
            processor_.join();
        }
        // END CLASS RUNTIME::CLEANER_

        // IMPLEMENTATION - STATICS
        inline
        void Runtime::process_tasks_()
        {
            std::vector<Task> t;
            do {
                t.clear();

                internal_stream_.try_dequeue_bulk(std::back_inserter(t), max_n_);
                for (auto&& x : t) {
                    bool poison{};
                    x(poison);

                    if (poison) return;
                }

                wait_all_streams_();

                // TODO: add backoff
                // if (std::empty(t)) {
                //     #if defined(YieldProcessor)
                //         YieldProcessor();
                //     #elif defined(__has_builtin)
                //         #if __has_builtin(__builtin_ia32_pause)
                //             __builtin_ia32_pause();
                //         #else
                //             std::this_thread::yield();
                //         #endif
                //     #else
                //         std::this_thread::yield();
                //     #endif
                // }
            } while (true);
        }

        inline
        void Runtime::wait_all_streams_()
        {
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
            return last_error_;
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

            return fut;
        }

        inline
        Stream* Runtime::null_stream()
        {
            static auto& r{streams_.emplace_front()};

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
            return std::exchange(last_error_, e);
        }

        inline
        void Runtime::synchronize()
        {   // TODO: redo, this induces ordering requirements on the processor.
            Task r{[](auto&&) { wait_all_streams_(); }};
            auto fut{r.get_future()};

            internal_stream_.enqueue(std::move(r));

            return fut.get();
        }
        // END CLASS RUNTIME
    } // Namespace hip::detail.
} // Namespace hip.