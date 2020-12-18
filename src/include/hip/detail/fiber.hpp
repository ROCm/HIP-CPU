/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable:4706)
#else
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wparentheses"
#endif
#if !defined(LIBCO_MP)
    #define LIBCO_MP
#endif
#if !defined(__clang__) && !defined(_MSC_VER)
    #define LIBCO_MPROTECT
#endif
#include "../../../../external/libco/libco.h"
#if defined(_MSC_VER)
    #pragma warning(pop)
#else
    #pragma GCC diagnostic pop
#endif

#if defined(thread_local) // libco sometimes squats on this space
    #undef thread_local
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace hip
{
    namespace detail
    {
        class Tile; // Forward declaration.
        class Tiled_domain; // Forward declaration.

        // BEGIN CLASS FIBER
        class Fiber final {
            // DATA - STATICS
            inline static constexpr std::uint32_t min_stack_byte_cnt_{
                (7168 + sizeof(void*) - 1) / sizeof(void*) * sizeof(void*)};
            inline static thread_local const Fiber* active_{nullptr};
            inline static thread_local std::uint32_t fiber_id_{0u};

            // DATA
            cothread_t f_{};
            std::uint32_t id_{};

            // FRIENDS
            friend class Tile;
            friend class Tiled_domain;

            // FRIENDS - COMPUTATIONAL BASIS
            friend
            inline
            bool operator==(const Fiber& x, const Fiber& y) noexcept
            {   // TODO: perhaps consider fiber id.
                return x.f_ == y.f_;
            }
            friend
            inline
            bool operator<(const Fiber& x, const Fiber& y) noexcept
            {   // TODO: consider fiber id.
                return x.f_ < y.f_;
            }

            // FRIENDS - ACCESSORS
            friend
            inline
            decltype(auto) id(const Fiber& x) noexcept
            {
                return x.id();
            }
            friend
            inline
            decltype(auto) native_handle(const Fiber& x) noexcept
            {
                return x.native_handle();
            }
            friend
            inline
            bool valid(const Fiber& x) noexcept { return x.valid(); }

            // IMPLEMENTATION - STATICS
            static
            Fiber& this_fiber_() noexcept;

            // IMPLEMENTATION - CREATORS
            explicit
            Fiber(cothread_t fiber) noexcept;
            template<
                typename F,
                std::enable_if_t<std::is_invocable_v<
                    decltype(co_create), std::uint32_t, F>>* = nullptr>
            Fiber(std::uint32_t stack_byte_cnt, F fn) noexcept;

            // IMPLEMENTATION - MANIPULATORS
            constexpr
            void set_id_(std::uint32_t new_id) noexcept;
        public:
            // NESTED TYPES
            using native_handle_type = decltype(f_);

            // STATICS
            static
            const Fiber& main() noexcept;
            static
            Fiber make(native_handle_type native_fiber) noexcept;
            template<typename F>
            static
            Fiber make(std::uint32_t stack_byte_cnt, F fn) noexcept;
            static
            const Fiber& this_fiber() noexcept;
            static
            void yield(const Fiber& to) noexcept;

            // CREATORS
            Fiber() = default;
            Fiber(const Fiber&) = default;
            Fiber(Fiber&&) = default;
            ~Fiber() = default;

            // MANIPULATORS
            Fiber& operator=(const Fiber&) = default;
            Fiber& operator=(Fiber&&) = default;

            // ACCESSORS
            std::uint32_t id() const noexcept;
            native_handle_type native_handle() const noexcept;
            bool valid() const noexcept;
            void yield_to(const Fiber& x) const noexcept;
        };

        // IMPLEMENTATION - STATICS
        inline
        Fiber& Fiber::this_fiber_() noexcept
        {
            if (!active_) active_ = &main();

            return *const_cast<Fiber*>(active_);
        }

        // IMPLEMENTATION - CREATORS
        inline
        Fiber::Fiber(cothread_t fiber) noexcept
            : f_{std::move(fiber)}, id_{fiber_id_++}
        {}

        template<
            typename F,
            std::enable_if_t<
                std::is_invocable_v<decltype(co_create), std::uint32_t, F>>*>
        inline
        Fiber::Fiber(std::uint32_t stack_byte_cnt, F fn) noexcept
            : f_{co_create(
                std::max(min_stack_byte_cnt_, stack_byte_cnt), std::move(fn))},
              id_{fiber_id_++}
        {}

        // IMPLEMENTATION - MANIPULATORS
        inline
        constexpr
        void Fiber::set_id_(std::uint32_t x) noexcept
        {
            id_ = x;
        }

        // STATICS
        inline
        const Fiber& Fiber::main() noexcept
        {
            static thread_local const Fiber r{co_active()};

            return r;
        }

        inline
        Fiber Fiber::make(native_handle_type native_fiber) noexcept
        {
            return Fiber{native_fiber};
        }

        template<typename F>
        inline
        Fiber Fiber::make(std::uint32_t stack_byte_cnt, F fn) noexcept
        {
            return Fiber{stack_byte_cnt, fn};
        }

        inline
        const Fiber& Fiber::this_fiber() noexcept
        {
            return this_fiber_();
        }

        inline
        void Fiber::yield(const Fiber& to) noexcept
        {
            active_ = &to;

            return co_switch(to.f_);
        }

        //ACCESSORS
        inline
        std::uint32_t Fiber::id() const noexcept
        {
            return id_;
        }

        inline
        typename Fiber::native_handle_type Fiber::native_handle() const noexcept
        {
            return f_;
        }

        inline
        bool Fiber::valid() const noexcept
        {
            return f_;
        }

        inline
        void Fiber::yield_to(const Fiber& x) const noexcept
        {
            return yield(x);
        }
        // END CLASS FIBER
    } // Namespace hip::detail.
} // Namespace hip.