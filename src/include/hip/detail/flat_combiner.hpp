/* -----------------------------------------------------------------------------
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "helpers.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <climits>
#include <cstdint>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hip
{
    namespace detail
    {
        // BEGIN CLASS FLAT_COMBINER
        template<typename T>
        class Flat_combiner final {
            // IMPLEMENTATION - NESTED TYPES
            struct Request_;
            struct Slot_;

            // DATA - STATICS
            inline thread_local static std::array<Slot_, UINT8_MAX> requests_{};

            // DATA
            T data_{};
            std::mutex mutex_{};
            std::atomic<Request_*> publication_list_{nullptr};

            // IMPLEMENTATION - MANIPULATORS
            void add_request_(Request_* ptr);
            bool try_combine_apply_();

            // IMPLEMENTATION - ACCESSORS
            Slot_& slot_for_this_thread_() const;
        public:
            using value_type = T;

            // CREATORS
            Flat_combiner() = default;
            explicit
            Flat_combiner(T data);
            template<
                typename... Args,
                std::enable_if_t<
                    (sizeof...(Args) > 0) &&
                    std::is_constructible_v<T, Args...>>* = nullptr>
            Flat_combiner(Args&&... ctor_args);
            Flat_combiner(const Flat_combiner&) = delete;
            Flat_combiner(Flat_combiner&&);
            ~Flat_combiner();

            // MANIPULATORS
            Flat_combiner& operator=(const Flat_combiner&) = delete;
            Flat_combiner& operator=(Flat_combiner&&) = default;
            template<typename UnaryProcedure>
            void apply(UnaryProcedure fn);

            // ACCESSORS
            operator const T&() const;
        };

        // IMPLEMENTATION - NESTED TYPES
        // BEGIN STRUCT FLAT_COMBINER::REQUEST_
        template<typename T>
        struct Flat_combiner<T>::Request_ {
            std::function<void (T&)> op{};
            Request_* next{};
            std::atomic<bool> done{false};
        };
        // END STRUCT FLAT_COMBINER::REQUEST_

        // BEGIN STRUCT FLAT_COMBINER::SLOT_
        template<typename T>
        struct Flat_combiner<T>::Slot_ {
            const void* data_ptr{};
            Request_* request_ptr{};

            // CREATORS
            Slot_() = default;
            Slot_(const Flat_combiner* this_ptr, Request_* request)
                : data_ptr{this_ptr}, request_ptr{request}
            {}
            Slot_(const Slot_&) = default;
            Slot_(Slot_&&) = default;
            ~Slot_() = default;
        };
        // END STRUCT FLAT_COMBINER::SLOT_

        // IMPLEMENTATION - MANIPULATORS
        template<typename T>
        inline
        void Flat_combiner<T>::add_request_(Request_* ptr)
        {
            auto old{publication_list_.load()};
            do {
                ptr->next = old;
            } while (!publication_list_.compare_exchange_weak(old, ptr));
        }

        template<typename T>
        inline
        bool Flat_combiner<T>::try_combine_apply_()
        {
            std::unique_lock<std::mutex> lck{mutex_, std::try_to_lock};

            if (!lck.owns_lock()) return false;

            auto it{publication_list_.exchange(nullptr)};
            while (it) {
                auto tmp{it->next};
                if (!it->done) {
                    it->op(data_);
                    it->done = true;
                }

                it = tmp;
            }

            return true;
        }

        // IMPLEMENTATION - ACCESSORS
        template<typename T>
        inline
        typename Flat_combiner<T>::Slot_&
            Flat_combiner<T>::slot_for_this_thread_() const
        {
            for (auto i = 0u; i != std::size(requests_); ++i) {
                if (requests_[i].data_ptr == this) return requests_[i];
                if (!requests_[i].data_ptr) {
                    requests_[i].data_ptr = this;
                    requests_[i].request_ptr = new Request_;

                    return requests_[i];
                }
            }

            throw std::runtime_error{"Overflowed combiner request array."};
        }

        // CREATORS
        template<typename T>
        inline
        Flat_combiner<T>::Flat_combiner(T x) : data_{std::move(x)} {}

        template<typename T>
        template<
            typename... Ts,
            std::enable_if_t<
            (sizeof...(Ts) > 0) && std::is_constructible_v<T, Ts...>>*>
        inline
        Flat_combiner<T>::Flat_combiner(Ts&&... xs)
            : data_{std::forward<Ts>(xs)...}
        {}

        template<typename T>
        inline
        Flat_combiner<T>::Flat_combiner(Flat_combiner&& x)
            : data_{std::move(x.data_)}
        {}

        template<typename T>
        inline
        Flat_combiner<T>::~Flat_combiner()
        {
            auto it{publication_list_.exchange(nullptr)};
            while (it) {
                auto tmp{it->next};
                delete it;

                it = tmp;
            }
        }

        // MANIPULATORS
        template<typename T>
        template<typename F>
        inline
        void Flat_combiner<T>::apply(F fn)
        {
            auto& tmp{slot_for_this_thread_()};

            tmp.request_ptr->op = std::move(fn);
            tmp.request_ptr->done = false;
            add_request_(tmp.request_ptr);

            std::uint8_t n{UINT8_MAX};
            do {
                if (tmp.request_ptr->done) return;
                if (try_combine_apply_()) return;

                if (n--) pause_or_yield();
            } while (true);
        }

        // ACCESSORS
        template<typename T>
        inline
        Flat_combiner<T>::operator const T&() const
        {
            return data_;
        }
        // END CLASS FLAT_COMBINER
    } // Namespace hip::detail.
} // Namespace hip.