/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "helpers.hpp"

#include <algorithm>
#include <array>
#include <execution>
#include <functional>
#include <version>

namespace hip
{
    namespace detail
    {
        // BEGIN STRUCT SCALAR_ACCESSOR
        template<typename T, typename Vector, std::uint32_t idx>
        struct Scalar_accessor final { // Idea from https://t0rakka.silvrback.com/simd-scalar-accessor
            // NESTED TYPES
            // BEGIN STRUCT ADDRESS
            struct Address {
                // DATA
                const Scalar_accessor* p;

                // MANIPULATORS
                operator T*() noexcept;
                operator T*() volatile noexcept;

                // ACCESSORS
                operator const T*() const noexcept;
                operator const T*() const volatile noexcept;
            };

            // DATA
            Vector data;

            // FRIENDS - MANIPULATORS
            friend
            inline
            std::istream& operator>>(
                std::istream& is, Scalar_accessor& x) noexcept
            {
                T tmp;
                is >> tmp;
                x.data[idx] = tmp;

                return is;
            }

            // FRIENDS - ACCESSORS
            friend
            inline
            std::ostream& operator<<(
                std::ostream& os, const Scalar_accessor& x) noexcept
            {
                return os << x.data[idx];
            }

            // CREATORS
            Scalar_accessor() = default;
            Scalar_accessor(const Scalar_accessor& x) noexcept;
            Scalar_accessor(Scalar_accessor&&) = default;
            ~Scalar_accessor() = default;

            // MANIPULATORS
            Scalar_accessor& operator=(const Scalar_accessor& x) noexcept;
            Scalar_accessor& operator=(T x) noexcept;
            volatile Scalar_accessor& operator=(T x) volatile noexcept;
            operator T&() noexcept;
            operator volatile T&() volatile noexcept;
            Scalar_accessor& operator++() noexcept;
            T operator++(int) noexcept;
            Scalar_accessor& operator--() noexcept;
            T operator--(int) noexcept;
            template< // TODO: constraint should actually check for the unary operator.
                typename U,
                std::enable_if_t<
                    std::is_invocable_v<std::plus<>, T, U>>* = nullptr>
            Scalar_accessor& operator+=(U x) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_invocable_v<std::minus<>, T, U>>* = nullptr>
            Scalar_accessor& operator-=(U x) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_invocable_v<std::multiplies<>, T, U>>* = nullptr>
            Scalar_accessor& operator*=(U x) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_invocable_v<std::divides<>, T, U>>* = nullptr>
            Scalar_accessor& operator/=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<
                    std::is_invocable_v<std::modulus<>, T, U>>* = nullptr>
            Scalar_accessor& operator%=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Scalar_accessor& operator>>=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Scalar_accessor& operator<<=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<
                    std::is_invocable_v<std::bit_and<>, T, U>>* = nullptr>
            Scalar_accessor& operator&=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<
                    std::is_invocable_v<std::bit_or<>, T, U>>* = nullptr>
            Scalar_accessor& operator|=(U x) noexcept;
            template<
                typename U = T,
                std::enable_if_t<
                    std::is_invocable_v<std::bit_xor<>, T, U>>* = nullptr>
            Scalar_accessor& operator^=(U x) noexcept;

            // ACCESSORS
            operator T() const noexcept;
            operator T() const volatile noexcept;
            Address operator&() const noexcept;
        };

        // NESTED TYPES
        // BEGIN STRUCT SCALAR_ACCESSOR::ADDRESS
        // MANIPULATORS
        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::Address::operator T*() noexcept
        {
            return &reinterpret_cast<T*>(const_cast<Scalar_accessor*>(p))[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::Address::operator T*() volatile noexcept
        {
            return &reinterpret_cast<T*>(const_cast<Scalar_accessor*>(p))[i];
        }

        // ACCESSORS
        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::Address::operator const T*() const noexcept
        {
            return &reinterpret_cast<const T*>(p)[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::Address::
            operator const T*() const volatile noexcept
        {
            return &reinterpret_cast<const T*>(p)[i];
        }
        // END STRUCT SCALAR_ACCESSOR::ADDRESS

        // CREATORS
        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::Scalar_accessor(
            const Scalar_accessor& x) noexcept : data{}
        {
            data[i] = x.data[i];
        }


        // MANIPULATORS
        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator=(
            const Scalar_accessor& x) noexcept
        {
            data[i] = x.data[i];

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator=(
            T x) noexcept
        {
            data[i] = x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        volatile Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator=(
            T x) volatile noexcept
        {
            data[i] = x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::operator T&() noexcept
        {
            return reinterpret_cast<T (&)[sizeof(V) / sizeof(T)]>(data)[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::operator volatile T&() volatile noexcept
        {
            return reinterpret_cast<
                volatile T (&)[sizeof(V) / sizeof(T)]>(data)[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::
            operator++() noexcept
        {
            ++data[i];

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        T Scalar_accessor<T, V, i>::operator++(int) noexcept
        {
            auto r{data[i]};
            ++data[i];

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::
            operator--() noexcept
        {
            --data[i];

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        T Scalar_accessor<T, V, i>::operator--(int) noexcept
        {
            auto r{data[i]};
            --data[i];

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
                std::enable_if_t<std::is_invocable_v<std::plus<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator+=(
            U x) noexcept
        {
            data[i] += x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::minus<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator-=(
            U x) noexcept
        {
            data[i] -= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::multiplies<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator*=(
            U x) noexcept
        {
            data[i] *= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::divides<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator/=(
            U x) noexcept
        {
            data[i] /= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::modulus<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator%=(
            U x) noexcept
        {
            data[i] %= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator>>=(
            U x) noexcept
        {
            data[i] >>= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator<<=(
            U x) noexcept
        {
            data[i] <<= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::bit_and<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator&=(
            U x) noexcept
        {
            data[i] &= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::bit_or<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator|=(
            U x) noexcept
        {
            data[i] |= x;

            return *this;
        }

        template<typename T, typename V, std::uint32_t i>
        template<
            typename U,
            std::enable_if_t<std::is_invocable_v<std::bit_xor<>, T, U>>*>
        inline
        Scalar_accessor<T, V, i>& Scalar_accessor<T, V, i>::operator^=(
            U x) noexcept
        {
            data[i] ^= x;

            return *this;
        }

        // ACCESSORS
        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::operator T() const noexcept
        {
            return data[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        Scalar_accessor<T, V, i>::operator T() const volatile noexcept
        {
            return data[i];
        }

        template<typename T, typename V, std::uint32_t i>
        inline
        typename Scalar_accessor<T, V, i>::Address Scalar_accessor<T, V, i>::
            operator&() const noexcept
        {
            return Address{this};
        }
        // END STRUCT SCALAR_ACCESSOR

        template<typename, std::uint32_t>
        struct Vector_base;

        template<typename T>
        struct Vector_base<T, 1> {
            // NESTED TYPES
            using V_ = std::array<T, 1>;

            // DATA
            union {
                V_ data_;
                Scalar_accessor<T, V_, 0> x;
            };
        };

        template<typename T>
        struct Vector_base<T, 2> {
            // NESTED TYPES
            using V_ = std::array<T, 2>;

            // DATA
            union {
                std::array<T, 2> data_;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
            };
        };

        template<typename T>
        struct Vector_base<T, 3> {
            // NESTED TYPES
            using V_ = std::array<T, 3>;

            // DATA
            union {
                V_ data_;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
                Scalar_accessor<T, V_, 2> z;
            };
        };

        template<typename T>
        struct Vector_base<T, 4> {
            // NESTED TYPES
            using V_ = std::array<T, 4>;

            // DATA
            union {
                V_ data_;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
                Scalar_accessor<T, V_, 2> z;
                Scalar_accessor<T, V_, 3> w;
            };
        };
        // END STRUCT VECTOR_BASE

        // BEGIN CLASS VECTOR_TYPE
        template<typename T, std::uint32_t rank>
        class Vector_type final : public Vector_base<T, rank> {
            // NESTED TYPES
            using B_ = Vector_base<T, rank>;
            using V_ = typename B_::V_;

            // DATA
            using B_::data_;

            // FRIENDS - ACCESSORS
            friend
            inline
            bool operator<(const Vector_type& x, const Vector_type& y) noexcept
            {
                return std::lexicographical_compare(
                    #if __cpp_lib_execution >= 201902L
                        std::execution::unseq,
                    #endif
                    std::cbegin(x.data_),
                    std::cend(x.data_),
                    std::cbegin(y.data_),
                    std::cend(y.data_));
            }
            friend
            inline
            bool operator==(const Vector_type& x, const Vector_type& y) noexcept
            {
                return std::equal(
                    #if __cpp_lib_execution >= 201902L
                        std::execution::unseq,
                    #endif
                    std::cbegin(x.data_),
                    std::cend(x.data_),
                    std::cbegin(y.data_));
            }
        public:
            // NESTED TYPES
            using value_type = T;

            // CREATORS
            constexpr
            Vector_type() noexcept;
            template<
                typename U,
                std::enable_if_t<std::is_constructible_v<T, U>>* = nullptr>
            constexpr
            explicit
            Vector_type(U t) noexcept;
            template< // TODO: constrain based on type as well.
                typename... Us,
                std::enable_if_t<
                    (rank > 1) &&
                    sizeof...(Us) == rank &&
                    std::conjunction_v<
                        std::is_constructible<T, Us>...>>* = nullptr>
            constexpr
            Vector_type(Us... xs) noexcept;
            constexpr
            Vector_type(const Vector_type& other) noexcept;
            constexpr
            Vector_type(Vector_type&& other) noexcept;
            ~Vector_type() = default;

            // MANIPULATORS
            Vector_type& operator=(const Vector_type& rhs) noexcept;
            Vector_type& operator=(Vector_type&& rhs) noexcept;
            Vector_type& operator++() noexcept;
            Vector_type operator++(int) noexcept;
            Vector_type& operator--() noexcept;
            Vector_type operator--(int) noexcept;
            Vector_type& operator+=(const Vector_type& addend) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            Vector_type& operator+=(U addend) noexcept;
            Vector_type& operator-=(const Vector_type& subtrahend) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            Vector_type& operator-=(U subtrahend) noexcept;
            Vector_type& operator*=(const Vector_type& multiplicand) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            Vector_type& operator*=(U multiplicand) noexcept;
            Vector_type& operator/=(const Vector_type& divisor) noexcept;
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            Vector_type& operator/=(U divisor) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator%=(const Vector_type& divisor) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator^=(const Vector_type& other) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator|=(const Vector_type& other) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator&=(const Vector_type& other) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator>>=(const Vector_type& shift) noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type& operator<<=(const Vector_type& shift) noexcept;

            // ACCESSORS
            Vector_type operator+() const noexcept;
            Vector_type operator-() const noexcept;
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            Vector_type operator~() const noexcept;
        };

        // CREATORS
        template<typename T, std::uint32_t n>
        inline
        constexpr
        Vector_type<T, n>::Vector_type() noexcept : B_{[]() { return V_{}; }()}
        {}

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_constructible_v<T, U>>*>
        inline
        constexpr
        Vector_type<T, n>::Vector_type(U t) noexcept
            : B_{[=]() noexcept { V_ r; r.fill(T(t)); return r; }()}
        {}

        template<typename T, std::uint32_t n>
        template< // TODO: constrain based on type as well.
            typename... Us,
            std::enable_if_t<
                (n > 1) &&
                sizeof...(Us) == n &&
                std::conjunction_v<std::is_constructible<T, Us>...>>*>
        inline
        constexpr
        Vector_type<T, n>::Vector_type(Us... xs) noexcept
            : B_{[=]() noexcept { return V_{T(xs)...}; }()}
        {}

        template<typename T, std::uint32_t n>
        inline
        constexpr
        Vector_type<T, n>::Vector_type(const Vector_type& t) noexcept
            : B_{t.data_}
        {}

        template<typename T, std::uint32_t n>
        inline
        constexpr
        Vector_type<T, n>::Vector_type(Vector_type&& t) noexcept
            : B_{std::move(t.data_)}
        {}

        // MANIPULATORS
        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator=(
            const Vector_type& t) noexcept
        {
            data_ = t.data_;

            return *this;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator=(
            Vector_type&& t) noexcept
        {
            data_ = std::move(t.data_);

            return *this;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator++() noexcept
        {
            for (auto&& t : data_) ++t;

            return *this;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> Vector_type<T, n>::operator++(int) noexcept
        {
            auto r{*this};
            ++*this;

            return r;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator--() noexcept
        {
            for (auto&& t : data_) --t;

            return *this;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> Vector_type<T, n>::operator--(int) noexcept
        {
            auto r{*this};
            --*this;

            return r;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator+=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::plus<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<
            typename U,
            std::enable_if_t<std::is_constructible_v<Vector_type<T, n>, U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator+=(U t) noexcept
        {
            return *this += Vector_type{std::move(t)};
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator-=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::minus<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<
            typename U,
            std::enable_if_t<std::is_constructible_v<Vector_type<T, n>, U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator-=(U t) noexcept
        {
            return *this -= Vector_type{std::move(t)};
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator*=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::multiplies<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<
            typename U,
            std::enable_if_t<std::is_constructible_v<Vector_type<T, n>, U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator*=(U t) noexcept
        {
            return *this *= Vector_type{std::move(t)};
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator/=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::divides<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<
            typename U,
            std::enable_if_t<std::is_constructible_v<Vector_type<T, n>, U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator/=(U t) noexcept
        {
            return *this /= Vector_type{std::move(t)};
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator%=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::modulus<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator^=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::bit_xor<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator|=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::bit_and<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<
            typename U,
            std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator&=(
            const Vector_type& t) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(t.data_),
                std::begin(data_),
                std::bit_and<>{});

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator>>=(
            const Vector_type& s) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(s.data_),
                std::begin(data_),
                [](auto&& t, auto && u) { return t >> u; });

            return *this;
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n>& Vector_type<T, n>::operator<<=(
            const Vector_type& s) noexcept
        {
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(data_),
                std::cend(data_),
                std::cbegin(s.data_),
                std::begin(data_),
                [](auto&& t, auto && u) { return t << u; });

            return *this;
        }

        // ACCESSORS
        template<typename T, std::uint32_t n>
        Vector_type<T, n> Vector_type<T, n>::operator+() const noexcept
        {
            return *this;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> Vector_type<T, n>::operator-() const noexcept
        {
            auto r{*this};
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(r.data_),
                std::cend(r.data_),
                std::begin(r.data_),
                std::negate<>{});

            return r;
        }

        template<typename T, std::uint32_t n>
        template<typename U, std::enable_if_t<std::is_integral_v<U>>*>
        inline
        Vector_type<T, n> Vector_type<T, n>::operator~() const noexcept
        {
            auto r{*this};
            std::transform(
                #if __cpp_lib_execution >= 201902L
                    std::execution::unseq,
                #endif
                std::cbegin(r.data_),
                std::cend(r.data_),
                std::begin(r.data_),
                std::bit_not<>{});

            return r;
        }
        // END CLASS VECTOR_TYPE

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> operator+(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} += y;
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator+(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} += Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator+(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} += y;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> operator-(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} -= y;
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator-(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} -= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator-(
            U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} -= y;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> operator*(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} *= y;
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator*(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} *= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator*(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} *= y;
        }

        template<typename T, std::uint32_t n>
        inline
        Vector_type<T, n> operator/(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} /= y;
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator/(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} /= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        Vector_type<T, n> operator/(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} /= y;
        }

        template<typename T, std::uint32_t n>
        inline
        bool operator<=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(y < x);
        }
        template<typename T, std::uint32_t n>
        inline
        bool operator>(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return y < x;
        }
        template<typename T, std::uint32_t n>
        inline
        bool operator >=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(x < y);
        }

        template<typename T, std::uint32_t n, typename U>
        inline
        bool operator==(const Vector_type<T, n>& x, U y) noexcept
        {
            return x == Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        bool operator==(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} == y;
        }

        template<typename T, std::uint32_t n>
        inline
        bool operator!=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(x == y);
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        bool operator!=(const Vector_type<T, n>& x, U y) noexcept
        {
            return !(x == y);
        }
        template<typename T, std::uint32_t n, typename U>
        inline
        bool operator!=(U x, const Vector_type<T, n>& y) noexcept
        {
            return !(x == y);
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator%(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} %= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator%(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} %= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator%(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} %= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator^(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} ^= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator^(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} ^= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator^(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} ^= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator|(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} |= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator|(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} |= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator|(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} |= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator&(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} &= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator&(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} &= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator&(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} &= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator>>(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} >>= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator>>(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} >>= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator>>(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} >>= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator<<(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} <<= y;
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator<<(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} <<= Vector_type<T, n>{y};
        }
        template<
            typename T,
            std::uint32_t n,
            typename U,
            std::enable_if_t<
                std::is_arithmetic_v<U> && std::is_integral_v<T>>* = nullptr>
        inline
        Vector_type<T, n> operator<<(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} <<= y;
        }
    } // Namespace hip::detail.
} // Namespace hip.