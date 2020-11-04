/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "stream.hpp"
#include "task.hpp"
#include "../../../../include/hip/hip_enums.h"

#include <algorithm>
#include <execution>
#include <forward_list>
#include <iterator>
#include <limits>
#include <thread>
#include <vector>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT HIP_RUNTIME
        struct Runtime final {
            inline static constexpr auto max_n{
                std::numeric_limits<Stream::ssize_t>::max()};
            inline static thread_local hipError_t last_error{};
            inline static Stream internal_stream{};
            inline static std::forward_list<Stream> streams{1};
            inline static Stream& null_stream{streams.front()};
            inline static std::thread processor{[]() {
                std::vector<Task> tmp;
                do {
                    tmp.clear();

                    internal_stream.try_dequeue_bulk(
                        std::back_inserter(tmp), max_n);
                    for (auto&& x : tmp) {
                        bool poison{};
                        x(poison);

                        if (poison) return;
                    }

                    std::for_each(
                        std::execution::par,
                        std::begin(streams),
                        std::end(streams),
                        [](auto&& x) {
                        static thread_local std::vector<Task> t;

                        x.try_dequeue_bulk(std::back_inserter(t), max_n);

                        for (auto&& y : t) { bool nop{}; y(nop); }

                        t.clear();
                    });

                    if (std::empty(tmp)) {
                        #if defined(YieldProcessor)
                            YieldProcessor();
                        #elif defined(__has_builtin)
                            #if __has_builtin(__builtin_ia32_pause)
                                __builtin_ia32_pause();
                            #else
                                std::this_thread::yield();
                            #endif
                        #else
                            std::this_thread::yield();
                        #endif
                    }
                } while (true);
            }};
            inline static const struct Cleaner {
                Cleaner() noexcept { last_error = hipSuccess; }
                ~Cleaner()
                {
                    internal_stream.enqueue(Task{[](auto&& p) { p = true; }});
                    processor.join();
                }
            } cleaner;

            // STATICS
            static
            std::future<void> destroy_stream_async(Stream* s)
            {
                Task r{[=](auto&&) {
                    streams.remove_if([=](auto&& x) { return &x == s; });
                }};
                auto fut{r.get_future()};

                internal_stream.enqueue(std::move(r));

                return fut;
            }

            static
            std::future<Stream*> make_stream_async()
            {
                auto p{new std::promise<Stream*>};
                auto fut{p->get_future()};

                internal_stream.enqueue(Task{[=](auto&&) mutable {
                    p->set_value(&streams.emplace_front());
                    delete p;
                }});

                return fut;
            }

            static
            std::future<void> synchronize()
            {   // TODO: redo, this induces ordering requirements on the processor.
                Task r{[](auto&&) {
                    std::for_each(
                        std::execution::par,
                        std::begin(streams),
                        std::end(streams),
                        [](auto&& x) {
                        static thread_local std::vector<Task> t;
                        x.try_dequeue_bulk(std::back_inserter(t), max_n);

                        for (auto&& y : t) { bool nop{}; y(nop); }

                        t.clear();
                    });
                }};
                auto fut{r.get_future()};

                internal_stream.enqueue(std::move(r));

                return fut;
            }
        };
        // END STRUCT HIP_RUNTIME
    } // Namespace hip::detail.
} // Namespace hip.