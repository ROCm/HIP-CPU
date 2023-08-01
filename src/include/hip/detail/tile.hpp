/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "coordinates.hpp"
#include "fiber.hpp"
#include "helpers.hpp"
#include "types.hpp"
#include "../../../../include/hip/hip_defines.h"
#include "../../../../include/hip/hip_constants.h"

#include <algorithm>
#include <execution>
#include <functional>
#include <cstdint>
#include <stdexcept>

namespace hip
{
    namespace detail
    {
        namespace this_tile
        {
            inline thread_local bool has_barrier{false};
        } // Namespace hip::detail::this_tile;

        class Tiled_domain; // Forward declaration.

        // BEGIN CLASS TILE
        class Tile final {
            // DATA
            Dim3 idx_{0, 0, 0};
            const Tiled_domain* domain_{};

            // FRIENDS
            friend class Tiled_domain;

            // FRIENDS - COMPUTATIONAL BASIS
            friend
            inline
            constexpr
            bool operator<(const Tile& x, const Tile& y) noexcept
            {
                return x.idx_ < y.idx_;
            }
            friend
            inline
            constexpr
            bool operator==(const Tile& x, const Tile& y) noexcept
            {
                return x.domain_ == y.domain_ && x.idx_ == y.idx_;
            }

            // FRIENDS - ACCESSORS
            friend
            inline
            void barrier(const Tile& tile) noexcept
            {
                return tile.barrier();
            }
            friend
            inline
            constexpr
            const Tiled_domain& domain(const Tile& tile) noexcept
            {
                return tile.domain();
            }
            friend
            inline
            constexpr
            const Dim3& dimensions(const Tile& x) noexcept
            {
                return x.dimensions();
            }
            friend
            inline
            constexpr
            const Dim3& index(const Tile& x) noexcept
            {
                return x.index();
            }

            // IMPLEMENTATION - STATICS
            static
            Tile& this_tile_() noexcept;
        public:
            // STATICS
            static
            decltype(auto) fibers() noexcept;
            template<
                typename F,
                typename... Args,
                std::enable_if_t<std::is_invocable_v<F, Args...>>* = nullptr>
            __HIP_FLATTENED_FUNCTION__
            static
            void for_each_fiber(
                const F& fn, const std::tuple<Args...>& args) noexcept;
            template<typename T, std::size_t n = warpSize>
            static
            decltype(auto) scratchpad() noexcept;
            static
            const Tile& this_tile() noexcept;

            // CREATORS
            constexpr
            Tile() noexcept = default;
            explicit
            constexpr
            Tile(
                const Tiled_domain& domain,
                const Dim3& tile_index = {0, 0, 0}) noexcept;
            constexpr
            Tile(const Tile&) noexcept = default;
            constexpr
            Tile(Tile&&) noexcept = default;
            ~Tile() = default;

            // MANIPULATORS
            Tile& operator=(const Tile&) noexcept = default;
            Tile& operator=(Tile&&) noexcept = default;

            // ACCESSORS
            __forceinline__
            void barrier() const noexcept;
            constexpr
            const Tiled_domain& domain() const noexcept;
            constexpr
            const Dim3& dimensions() const noexcept;
            constexpr
            const Dim3& index() const noexcept;
        };

        // IMPLEMENTATION - STATICS
        inline
        Tile& Tile::this_tile_() noexcept
        {
            thread_local static Tile r{};

            return r;
        }

        // STATICS
        template<
            typename F,
            typename... Args,
            std::enable_if_t<std::is_invocable_v<F, Args...>>*>
        inline
        void Tile::for_each_fiber(
            const F& fn, const std::tuple<Args...>& args) noexcept
        {
            __HIP_VECTORISED_LOOP__
            for (auto i = 0u; i < count(this_tile().dimensions()); ++i) {
                Fiber::this_fiber_().set_id_(i);

                std::apply(fn, args);
            }

            Fiber::this_fiber_().set_id_(0);
        }

        template<typename T, std::size_t n>
        inline
        decltype(auto) Tile::scratchpad() noexcept
        {   // TODO: use named variable for maximum block size.
            thread_local static T r[1024 / warpSize][n];

            const auto widx{id(hip::detail::Fiber::this_fiber()) / warpSize};

            return (r[widx]);
        }

        inline
        const Tile& Tile::this_tile() noexcept
        {
            return this_tile_();
        }

        // CREATORS
        inline
        constexpr
        Tile::Tile(const Tiled_domain& d, const Dim3& tidx) noexcept
            : idx_{tidx}, domain_{&d}
        {}

        // ACCESSORS
        inline
        constexpr
        const Tiled_domain& Tile::domain() const noexcept
        {
            return *domain_;
        }

        inline
        constexpr
        const Dim3& Tile::index() const noexcept
        {
            return idx_;
        }
        // END CLASS TILE

        // BEGIN CLASS TILED_DOMAIN
        class Tiled_domain final {
            // NESTED TYPES
            // BEGIN CLASS TILED_DOMAIN::ITERATOR_
            class Iterator_ final {
                // DATA
                const Tiled_domain* domain_{};
                std::uint32_t idx_{};

                // FRIENDS - COMPUTATIONAL BASIS
                friend
                inline
                Iterator_ operator+(
                    const Iterator_& x, std::int32_t dx) noexcept
                {
                    return Iterator_{x} += dx;
                }
                friend
                inline
                std::int32_t operator-(
                    const Iterator_& x, const Iterator_& y) noexcept
                {   // TODO: redo.
                    return bit_cast<std::int32_t>(x.idx_ - y.idx_);
                }
                friend
                inline
                constexpr
                bool operator==(const Iterator_& x, const Iterator_& y) noexcept
                {
                    return x.domain_ == y.domain_ && x.idx_ == y.idx_;
                }
                friend
                inline
                constexpr
                bool operator!=(const Iterator_& x, const Iterator_& y) noexcept
                {
                    return !(x == y);
                }
                friend
                inline
                constexpr
                bool operator<(const Iterator_& x, const Iterator_& y) noexcept
                {   // TODO: should this consider domain_?
                    return x.idx_ < y.idx_;
                }
            public:
                // NESTED TYPES
                using difference_type = std::int32_t;
                using iterator_category = std::random_access_iterator_tag;
                using pointer = void;
                using reference = void;
                using value_type = Tile;

                // CREATORS
                Iterator_() = default;
                explicit
                constexpr
                Iterator_(
                    const Tiled_domain& domain,
                    std::uint32_t index = 0u) noexcept;
                Iterator_(const Iterator_&) = default;
                Iterator_(Iterator_&&) = default;
                ~Iterator_() = default;

                // MANIPULATORS
                Iterator_& operator=(const Iterator_&) = default;
                Iterator_& operator=(Iterator_&&) = default;
                Iterator_& operator++() noexcept;
                Iterator_ operator++(int) noexcept;
                Iterator_& operator+=(difference_type dx) noexcept;
                Iterator_& operator--() noexcept;
                Iterator_ operator--(int) noexcept;
                Iterator_& operator-=(difference_type dx) noexcept;
                value_type operator*() noexcept;
                value_type operator[](difference_type dx) noexcept;

                // ACCESSORS
                value_type operator*() const noexcept;
                value_type operator[](difference_type dx) const noexcept;
            };
            // END CLASS TILED_DOMAIN::ITERATOR_
            using Kernel_ = std::function<void ()>;

            // DATA
            Kernel_ kernel_{};
            Dim3 tile_cnt_{};
            Dim3 tile_dim_{};
            std::uint32_t scratch_byte_cnt_{};
            std::uint32_t size_{};

            // FRIENDS - ACCESSORS
            template<typename F, typename... Args>
            friend
            inline
            void for_each_tile(
                const Tiled_domain& x,
                const F& fiber_fn,
                const std::tuple<Args...>& args) noexcept
            {
                if (count(x.tile_dimensions()) == 1) {
                    return x.for_each_unit_tile_(fiber_fn, args);
                }

                return x.for_each_tile_(fiber_fn, args);
            }
            friend
            inline
            constexpr
            const Kernel_& kernel(const Tiled_domain& x) noexcept
            {
                return x.kernel();
            }
            friend
            inline
            constexpr
            std::uint32_t scratchpad_size(const Tiled_domain& x) noexcept
            {
                return x.scratchpad_size();
            }
            friend
            inline
            constexpr
            const Dim3& tile_count(const Tiled_domain& x) noexcept
            {
                return x.tile_count();
            }
            friend
            inline
            constexpr
            const Dim3& tile_dimensions(const Tiled_domain& x) noexcept
            {
                return x.tile_dimensions();
            }

            // IMPLEMENTATION - STATICS
            template<typename F>
            static
            decltype(auto) make_tile_fn_(F fn) noexcept;

            // IMPLEMENTATION - ACCESSORS
            template<typename F, typename... Args>
            __HIP_FLATTENED_FUNCTION__
            void for_each_tile_(
                const F& fn, const std::tuple<Args...>& args) const noexcept;
            template<typename F, typename... Args>
            __HIP_FLATTENED_FUNCTION__
            void for_each_unit_tile_(
                const F& fn, const std::tuple<Args...>& args) const noexcept;
        public:
            using const_iterator = Iterator_;
            using iterator = Iterator_;
            using kernel_type = Kernel_;
            using size_type = std::uint32_t;

            // CREATORS
            Tiled_domain() = default;
            template<
                typename F,
                std::enable_if_t<
                    std::is_constructible_v<kernel_type, F>>* = nullptr>
            Tiled_domain(
                const Dim3& tile_dim,
                const Dim3& tile_count,
                std::uint32_t scratchpad_byte_cnt,
                F global_fn) noexcept;
            Tiled_domain(const Tiled_domain&) = default;
            Tiled_domain(Tiled_domain&&) = default;
            ~Tiled_domain() = default;

            // MANIPULATORS
            Tiled_domain& operator=(const Tiled_domain&) = default;
            Tiled_domain& operator=(Tiled_domain&&) = default;
            iterator begin() noexcept;
            iterator end() noexcept;

            // ACCESSORS
            constexpr
            const_iterator begin() const noexcept;
            constexpr
            const_iterator cbegin() const noexcept;
            constexpr
            const_iterator cend() const noexcept;
            constexpr
            const_iterator end() const noexcept;
            constexpr
            const kernel_type& kernel() const noexcept;
            constexpr
            std::uint32_t scratchpad_size() const noexcept;
            constexpr
            size_type size() const noexcept;
            constexpr
            const Dim3& tile_count() const noexcept;
            constexpr
            const Dim3& tile_dimensions() const noexcept;
        };

        // BEGIN DEPENDENT MEMBERS OF CLASS TILE
        // STATICS
        inline
        decltype(auto) Tile::fibers() noexcept
        {
            static thread_local std::vector<Fiber> r{Fiber::main()};
            struct D {
                D() noexcept
                {
                    static constexpr auto max_fiber_cnt{1024u};

                    r.reserve(max_fiber_cnt);
                }
                ~D()
                {
                    for (auto i = 1u; i != std::size(r); ++i) {
                        co_delete(native_handle(r[i]));
                    }
                }
            } static thread_local const deleter{};

            while (std::size(r) < count(this_tile().dimensions())) {
                r.push_back(Fiber::make(256, []() {
                    while (true) {
                        this_tile().domain().kernel()();

                        const auto f1{id(Fiber::this_fiber())};
                        const auto f2{
                            (f1 + 1) % count(this_tile().dimensions())};

                        Fiber::yield(r[f2]);
                    }
                }));
            }

            return static_cast<const std::vector<Fiber>&>(r);
        }

        // ACCESSORS
        inline
        void Tile::barrier() const noexcept
        {
            if (count(dimensions()) == 1) return;

            hip::detail::this_tile::has_barrier = true;

            const auto f0{id(Fiber::this_fiber())};
            const auto f1{(f0 + 1) % count(dimensions())};

            Fiber::yield(Tile::fibers()[f1]);
        }

        inline
        constexpr
        const Dim3& Tile::dimensions() const noexcept
        {
            return domain_->tile_dimensions();
        }
        // END DEPENDENT MEMBERS OF CLASS TILE

        // NESTED TYPES
        // BEGIN CLASS TILED_DOMAIN::ITERATOR_
        // CREATORS
        inline
        constexpr
        Tiled_domain::Iterator_::Iterator_(
            const Tiled_domain& domain, std::uint32_t index) noexcept
            : domain_{&domain}, idx_{index}
        {}

        // MANIPULATORS
        inline
        Tiled_domain::Iterator_& Tiled_domain::Iterator_::operator++() noexcept
        {
            ++idx_;

            return *this;
        }

        inline
        Tiled_domain::Iterator_ Tiled_domain::Iterator_::operator++(
            int) noexcept
        {
            auto tmp{*this};
            ++*this;

            return tmp;
        }

        inline
        Tiled_domain::Iterator_& Tiled_domain::Iterator_::operator+=(
            difference_type dx) noexcept
        {
            idx_ += dx;

            return *this;
        }

        inline
        Tiled_domain::Iterator_& Tiled_domain::Iterator_::operator--() noexcept
        {
            --idx_;

            return *this;
        }

        inline
        Tiled_domain::Iterator_ Tiled_domain::Iterator_::operator--(
            int) noexcept
        {
            auto tmp{*this};
            --*this;

            return tmp;
        }

        inline
        Tiled_domain::Iterator_& Tiled_domain::Iterator_::operator-=(
            difference_type dx) noexcept
        {
            idx_ -= dx;

            return *this;
        }

        inline
        typename Tiled_domain::Iterator_::value_type
            Tiled_domain::Iterator_::operator*() noexcept
        {
            return value_type{*domain_, extrude(domain_->tile_count(), idx_)};
        }

        inline
        typename Tiled_domain::Iterator_::value_type
            Tiled_domain::Iterator_::operator[](difference_type dx) noexcept
        {
            return *(*this + dx);
        }

        // ACCESSORS
        inline
        typename Tiled_domain::Iterator_::value_type
            Tiled_domain::Iterator_::operator*() const noexcept
        {
            return value_type{*domain_, extrude(domain_->tile_count(), idx_)};
        }

        inline
        typename Tiled_domain::Iterator_::value_type Tiled_domain::Iterator_::
            operator[](difference_type dx) const noexcept
        {
            return *(*this + dx);
        }
        // END CLASS TILED_DOMAIN::ITERATOR_

        // IMPLEMENTATION - STATICS
        template<typename F>
        inline
        decltype(auto) Tiled_domain::make_tile_fn_(F fn) noexcept
        {
            struct {
                F fn_;

                __HIP_TILE_FUNCTION__
                void operator()(Tile&& tile) const noexcept
                {
                    return
                        std::apply(fn_, std::forward_as_tuple(std::move(tile)));
                }
            } r{std::move(fn)};

            return r;
        }

        // IMPLEMENTATION - ACCESSORS
        template<typename F, typename... Args>
        inline
        void Tiled_domain::for_each_tile_(
            const F& fn, const std::tuple<Args...>& args) const noexcept
        {
            std::for_each(
                std::execution::par_unseq,
                cbegin(),
                cend(),
                make_tile_fn_([&](auto&& tile) noexcept {
                Tile::this_tile_() = std::move(tile);
                this_tile::has_barrier = false;

                std::apply(fn, args);

                if (this_tile::has_barrier) {
                    Fiber::yield(Tile::fibers()[1]);
                } else {
                    __HIP_VECTORISED_LOOP__
                    for (auto i = 1u; i < count(tile_dimensions()); ++i) {
                        Fiber::this_fiber_().set_id_(i);

                        std::apply(fn, args);
                    }
                }

                Fiber::this_fiber_().set_id_(0u);
            }));
        }

        template<typename F, typename... Args>
        inline
        void Tiled_domain::for_each_unit_tile_(
            const F& fn, const std::tuple<Args...>& args) const noexcept
        {
            std::for_each_n(
                std::execution::par_unseq,
                cbegin(),
                size(),
                make_tile_fn_([&](auto&& tile) noexcept {
                Tile::this_tile_() = std::move(tile);

                std::apply(fn, args);
            }));
        }

        // CREATORS
        template<
            typename F,
            std::enable_if_t<
                std::is_constructible_v<typename Tiled_domain::kernel_type, F>>*>
        inline
        Tiled_domain::Tiled_domain(
            const Dim3& td,
            const Dim3& t_cnt,
            std::uint32_t scratch_bytes,
            F fn) noexcept
            :
            kernel_{std::move(fn)},
            tile_cnt_{t_cnt},
            tile_dim_{td},
            scratch_byte_cnt_{scratch_bytes},
            size_{count(td) * count(t_cnt)}
        {}

        // MANIPULATORS
        inline
        typename Tiled_domain::iterator Tiled_domain::begin() noexcept
        {
            return iterator{*this};
        }

        inline
        typename Tiled_domain::iterator Tiled_domain::end() noexcept
        {
            return iterator{*this, count(tile_count())};
        }

        // ACCESSORS
        inline
        constexpr
        typename Tiled_domain::const_iterator
            Tiled_domain::begin() const noexcept
        {
            return iterator{*this};
        }

        inline
        constexpr
        typename Tiled_domain::const_iterator
            Tiled_domain::cbegin() const noexcept
        {
            return begin();
        }

        inline
        constexpr
        typename Tiled_domain::const_iterator
            Tiled_domain::cend() const noexcept
        {
            return end();
        }

        inline
        constexpr
        typename Tiled_domain::const_iterator Tiled_domain::end() const noexcept
        {
            return iterator{*this, count(tile_count())};
        }

        inline
        constexpr
        const std::function<void ()>& Tiled_domain::kernel() const noexcept
        {
            return kernel_;
        }

        inline
        constexpr
        std::uint32_t Tiled_domain::scratchpad_size() const noexcept
        {
            return scratch_byte_cnt_;
        }

        inline
        constexpr
        typename Tiled_domain::size_type Tiled_domain::size() const noexcept
        {
            return size_;
        }

        inline
        constexpr
        const Dim3& Tiled_domain::tile_count() const noexcept
        {
            return tile_cnt_;
        }

        inline
        constexpr
        const Dim3& Tiled_domain::tile_dimensions() const noexcept
        {
            return tile_dim_;
        }
        // END CLASS TILED_DOMAIN

        namespace this_tile
        {
            extern "C"
            {
                __HIP_API__
                inline
                void* _hip_detail_this_tile_so_local_context() noexcept
                {
                    return const_cast<Tile*>(&Tile::this_tile());
                }
            } // extern "C".
        } // Namespace hip::detail::this_tile.
    } // Namespace hip::detail.
} // Namespace hip.