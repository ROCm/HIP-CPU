/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "helpers.hpp"
#include "../../../../include/hip/hip_defines.h"

#if defined(_MSC_VER)
    #pragma warning(push, 0)
#else
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-compare"
    #pragma GCC diagnostic ignored "-Wunused"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "../../../../external/half/half.hpp"
#include "../../../../external/ssm/vector.hpp"
#if defined(_MSC_VER)
    #pragma warning(pop)
#else
    #pragma GCC diagnostic pop
#endif

#include <array>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace hip
{
    namespace detail
    {
        template<typename, typename, std::uint32_t> struct Scalar_accessor;
    } // Namespace hip::detail.
} // Namespace hip.

namespace std
{
    template<typename T, typename U, uint32_t n>
    struct is_integral<hip::detail::Scalar_accessor<T, U, n>>
        : is_integral<T> {};
    template<typename T, typename U, uint32_t n>
    struct is_floating_point<hip::detail::Scalar_accessor<T, U, n>>
        : is_floating_point<T> {};
} // Namespace std.

// TODO: temporary workaround, probably preferable to fix in ssm.
namespace ssm
{
    namespace simd
    {
        template<>
        struct vector_data<half_float::half, 2> {
            struct Trivial_half {
                std::aligned_storage_t<
                    sizeof(half_float::half), alignof(half_float::half)> x;

                operator half_float::half() const noexcept
                {
                    return hip::detail::bit_cast<half_float::half>(x);
                }
            };

            union {
                accessor<Trivial_half, 2, 0> x;
                accessor<Trivial_half, 2, 1> y;
                vector<half_float::half, 2> data = {};
            };
        };
    } // Namespace ssm::simd.
} // Namespace ssm.

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
                __host__ __device__
                operator T*() noexcept
                {
                    return &reinterpret_cast<T*>(
                        const_cast<Scalar_accessor*>(p))[idx];
                }
                __host__ __device__
                operator T*() volatile noexcept
                {
                    return &reinterpret_cast<T*>(
                        const_cast<Scalar_accessor*>(p))[idx];
                }

                // ACCESSORS
                __host__ __device__
                operator const T*() const noexcept
                {
                    return &reinterpret_cast<const T*>(p)[idx];
                }
                __host__ __device__
                operator const T*() const volatile noexcept
                {
                    return &reinterpret_cast<const T*>(p)[idx];
                }
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
            __host__ __device__
            Scalar_accessor() = default;
            __host__ __device__
            Scalar_accessor(const Scalar_accessor& x) noexcept : data{}
            {
                data[idx] = x.data[idx];
            }
            Scalar_accessor(Scalar_accessor&&) = default;
            ~Scalar_accessor() = default;

            // ACCESSORS
            __host__ __device__
            operator T() const noexcept { return data[idx]; }
            __host__ __device__
            operator T() const volatile noexcept { return data[idx]; }
            __host__ __device__
            Address operator&() const noexcept { return Address{this}; }

            // MANIPULATORS
            __host__ __device__
            Scalar_accessor& operator=(const Scalar_accessor& x) noexcept
            {
                data[idx] = x.data[idx];

                return *this;
            }
            __host__ __device__
            Scalar_accessor& operator=(T x) noexcept
            {
                data[idx] = x;

                return *this;
            }
            __host__ __device__
            volatile Scalar_accessor& operator=(T x) volatile noexcept
            {
                data[idx] = x;

                return *this;
            }

            __host__ __device__
            operator T&() noexcept
            {
                return reinterpret_cast<
                    T (&)[sizeof(Vector) / sizeof(T)]>(data)[idx];
            }
            __host__ __device__
            operator volatile T&() volatile noexcept
            {
                return reinterpret_cast<
                    volatile T (&)[sizeof(Vector) / sizeof(T)]>(data)[idx];
            }

            __host__ __device__
            Scalar_accessor& operator++() noexcept
            {
                ++data[idx];

                return *this;
            }
            __host__ __device__
            T operator++(int) noexcept
            {
                auto r{data[idx]};
                ++data[idx];

                return *this;
            }

            __host__ __device__
            Scalar_accessor& operator--() noexcept
            {
                --data[idx];

                return *this;
            }
            __host__ __device__
            T operator--(int) noexcept
            {
                auto r{data[idx]};
                --data[idx];

                return *this;
            }

            // TODO: constraint should actually check for the unary operator.
            template<
                typename U,
                std::enable_if_t<std::is_invocable_v<
                    std::plus<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator+=(U x) noexcept
            {
                data[idx] += x;

                return *this;
            }
            template<
                typename U,
                std::enable_if_t<std::is_invocable_v<
                    std::minus<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator-=(U x) noexcept
            {
                data[idx] -= x;

                return *this;
            }

            template<
                typename U,
                std::enable_if_t<std::is_invocable_v<
                    std::multiplies<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator*=(U x) noexcept
            {
                data[idx] *= x;

                return *this;
            }
            template<
                typename U,
                std::enable_if_t<std::is_invocable_v<
                    std::divides<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator/=(U x) noexcept
            {
                data[idx] /= x;

                return *this;
            }
            template<
                typename U = T,
                std::enable_if_t<std::is_invocable_v<
                    std::modulus<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator%=(U x) noexcept
            {
                data[idx] %= x;

                return *this;
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator>>=(U x) noexcept
            {
                data[idx] >>= x;

                return *this;
            }
            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator<<=(U x) noexcept
            {
                data[idx] <<= x;

                return *this;
            }
            template<
                typename U = T,
                std::enable_if_t<std::is_invocable_v<
                    std::bit_and<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator&=(U x) noexcept
            {
                data[idx] &= x;

                return *this;
            }
            template<
                typename U = T,
                std::enable_if_t<std::is_invocable_v<
                    std::bit_or<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator|=(U x) noexcept
            {
                data[idx] |= x;

                return *this;
            }
            template<
                typename U = T,
                std::enable_if_t<std::is_invocable_v<
                    std::bit_xor<>, decltype(data[idx]), U>>* = nullptr>
            __host__ __device__
            Scalar_accessor& operator^=(U x) noexcept
            {
                data[idx] ^= x;

                return *this;
            }
        };
        // END STRUCT SCALAR_ACCESSOR

        // BEGIN STRUCT VECTOR_BASE
        template<typename T, std::uint32_t n> struct Vector_base;

        template<typename T>
        struct Vector_base<T, 1> {
            // NESTED TYPES
            using V_ = ssm::vector<T, 1>;
            using value_type = T;

            // DATA
            union {
                V_ data;
                Scalar_accessor<T, V_, 0> x;
            };

            // CREATORS
            constexpr
            Vector_base() noexcept : data{} {}
            constexpr
            explicit
            Vector_base(T xx) noexcept : data{xx} {}
            constexpr
            Vector_base(const Vector_base& xx) noexcept : data{xx.data} {}
            constexpr
            Vector_base(Vector_base&& xx) noexcept : data{std::move(xx.data)} {}
            ~Vector_base() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_base& operator=(const Vector_base& xx) noexcept
            {
                data = xx.data;

                return *this;
            }
            __host__ __device__
            Vector_base& operator=(Vector_base&& xx) noexcept
            {
                data = std::move(xx.data);

                return *this;
            }
        };

        template<typename T>
        struct Vector_base<T, 2> {
            // NESTED TYPES
            using V_ = ssm::vector<T, 2>;
            using value_type = T;

            // DATA
            union {
                V_ data;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
            };

            // CREATORS
            constexpr
            Vector_base() noexcept : data{} {}
            constexpr
            explicit
            Vector_base(T xx) noexcept : data{xx} {}
            constexpr
            explicit
            Vector_base(T xx, T yy) noexcept : data{xx, yy} {}
            constexpr
            Vector_base(const Vector_base& xx) noexcept : data{xx.data} {}
            constexpr
            Vector_base(Vector_base&& xx) noexcept : data{std::move(xx.data)} {}
            ~Vector_base() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_base& operator=(const Vector_base& xx) noexcept
            {
                data = xx.data;

                return *this;
            }
            __host__ __device__
            Vector_base& operator=(Vector_base&& xx) noexcept
            {
                data = std::move(xx.data);

                return *this;
            }
        };

        // half_float::half is non-trivial, hence it's easier to specialise for
        // it
        template<>
        struct Vector_base<half_float::half, 2> {
            // NESTED TYPES
            using V_ = ssm::vector<half_float::half, 2>;
            using value_type = half_float::half;

            // DATA
            union {
                V_ data;
                Scalar_accessor<half_float::half, V_, 0> x;
                Scalar_accessor<half_float::half, V_, 1> y;
            };

            // CREATORS
            Vector_base() noexcept : data{value_type{0}} {}
            explicit
            Vector_base(value_type xx) noexcept : data{xx} {}
            explicit
            Vector_base(value_type xx, value_type yy) noexcept : data{xx, yy} {}
            Vector_base(const Vector_base& xx) noexcept : data{xx.data} {}
            Vector_base(Vector_base&& xx) noexcept : data{std::move(xx.data)} {}
            ~Vector_base() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_base& operator=(const Vector_base& xx) noexcept
            {
                data = xx.data;

                return *this;
            }
            __host__ __device__
            Vector_base& operator=(Vector_base&& xx) noexcept
            {
                data = std::move(xx.data);

                return *this;
            }
        };

        template<typename T>
        struct Vector_base<T, 3> {
            // NESTED TYPES
            using V_ = ssm::vector<T, 3>;
            using value_type = T;

            // DATA
            union {
                V_ data;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
                Scalar_accessor<T, V_, 2> z;
            };

            // CREATORS
            constexpr
            Vector_base() noexcept : data{} {}
            constexpr
            explicit
            Vector_base(T xx) noexcept : data{xx} {}
            constexpr
            explicit
            Vector_base(T xx, T yy, T zz) noexcept : data{xx, yy, zz} {}
            constexpr
            Vector_base(const Vector_base& xx) noexcept : data{xx.data} {}
            constexpr
            Vector_base(Vector_base&& xx) noexcept : data{std::move(xx.data)} {}
            ~Vector_base() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_base& operator=(const Vector_base& xx) noexcept
            {
                data = xx.data;

                return *this;
            }
            __host__ __device__
            Vector_base& operator=(Vector_base&& xx) noexcept
            {
                data = std::move(xx.data);

                return *this;
            }
        };

        template<typename T>
        struct Vector_base<T, 4> {
            // NESTED TYPES
            using V_ = ssm::vector<T, 4>;
            using value_type = T;

            // DATA
            union {
                V_ data;
                Scalar_accessor<T, V_, 0> x;
                Scalar_accessor<T, V_, 1> y;
                Scalar_accessor<T, V_, 2> z;
                Scalar_accessor<T, V_, 3> w;
            };

            // CREATORS
            constexpr
            Vector_base() noexcept : data{} {}
            constexpr
            explicit
            Vector_base(T xx) noexcept : data{xx} {}
            constexpr
            explicit
            Vector_base(T xx, T yy, T zz, T ww) noexcept
                : data{xx, yy, zz, ww}
            {}
            constexpr
            Vector_base(const Vector_base& xx) noexcept : data{xx.data} {}
            constexpr
            Vector_base(Vector_base&& xx) noexcept : data{std::move(xx.data)} {}
            ~Vector_base() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_base& operator=(const Vector_base& xx) noexcept
            {
                data = xx.data;

                return *this;
            }
            __host__ __device__
            Vector_base& operator=(Vector_base&& xx) noexcept
            {
                data = std::move(xx.data);

                return *this;
            }
        };
        // END STRUCT VECTOR_BASE

        // BEGIN STRUCT VECTOR_TYPE
        template<typename T, std::uint32_t rank>
        struct Vector_type final : public Vector_base<T, rank> {
            // NESTED TYPES
            using Vector_base<T, rank>::data;
            using typename Vector_base<T, rank>::V_;

            // FRIENDS - ACCESSORS
            friend
            inline
            __host__ __device__
            bool operator<(const Vector_type& x, const Vector_type& y) noexcept
            {
                if constexpr (rank == 1) return x.x < y.x;
                else if constexpr (rank == 2) return x.x < y.x && x.y < y.y;
                else if constexpr (rank == 3) {
                    return x.x < y.x && x.y < y.y && x.z < y.z;
                }
                else return x.x < y.x && x.y < y.y && x.z < y.z && x.w < y.w;

                return false;
            }
            friend
            inline
            __host__ __device__
            bool operator==(const Vector_type& x, const Vector_type& y) noexcept
            {
                return x.data == y.data;
            }

            // CREATORS
            __host__ __device__
            constexpr
            Vector_type() noexcept = default;
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<
                        Vector_base<T, rank>,
                        decltype(static_cast<T>(std::declval<U>()))>>* = nullptr>
            __host__ __device__
            constexpr
            explicit
            Vector_type(U xx) noexcept
                : Vector_base<T, rank>{static_cast<T>(xx)}
            {}
            template< // TODO: constrain based on type as well.
                typename... Us,
                std::enable_if_t<
                    (rank > 1) &&
                    sizeof...(Us) == rank/* &&
                    std::is_constructible_v<
                        Vector_base<T, rank>,
                        decltype(static_cast<T>(std::declval<Us>()))...>*/>* = nullptr>
            __host__ __device__
            constexpr
            Vector_type(Us... xs) noexcept
                : Vector_base<T, rank>{static_cast<T>(xs)...}
            {}
            __host__ __device__
            constexpr
            Vector_type(const Vector_type&) noexcept = default;
            __host__ __device__
            constexpr
            Vector_type(Vector_type&&) noexcept = default;
            __host__ __device__
            ~Vector_type() = default;

            // MANIPULATORS
            __host__ __device__
            Vector_type& operator=(const Vector_type&) noexcept = default;
            __host__ __device__
            Vector_type& operator=(Vector_type&&) noexcept = default;

            __host__ __device__
            Vector_type& operator++() noexcept
            {
                data += V_{static_cast<T>(1)};

                return *this;
            }
            __host__ __device__
            Vector_type operator++(int) noexcept
            {
                auto tmp(*this);
                ++*this;

                return tmp;
            }

            __host__ __device__
            Vector_type& operator--() noexcept
            {
                data -= V_{static_cast<T>(1)};

                return *this;
            }
            __host__ __device__
            Vector_type operator--(int) noexcept
            {
                auto tmp(*this);
                --*this;

                return tmp;
            }

            __host__ __device__
            Vector_type& operator+=(const Vector_type& xx) noexcept
            {
                data += xx.data;

                return *this;
            }
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            __host__ __device__
            Vector_type& operator+=(U xx) noexcept
            {
                return *this += Vector_type{xx};
            }

            __host__ __device__
            Vector_type& operator-=(const Vector_type& xx) noexcept
            {
                data -= xx.data;
                return *this;
            }
            template<
                typename U,
                std::enable_if_t<
                std::is_constructible_v<Vector_type, U>>* = nullptr>
            __host__ __device__
            Vector_type& operator-=(U xx) noexcept
            {
                return *this -= Vector_type{xx};
            }

            __host__ __device__
            Vector_type& operator*=(const Vector_type& xx) noexcept
            {
                data *= xx.data;

                return *this;
            }
            template<
                typename U,
                std::enable_if_t<
                std::is_constructible_v<Vector_type, U>>* = nullptr>
            __host__ __device__
            Vector_type& operator*=(U xx) noexcept
            {
                return *this *= Vector_type{xx};
            }

            __host__ __device__
            Vector_type& operator/=(const Vector_type& xx) noexcept
            {
                data /= xx.data;

                return *this;
            }
            template<
                typename U,
                std::enable_if_t<
                    std::is_constructible_v<Vector_type, U>>* = nullptr>
            __host__ __device__
            Vector_type& operator/=(U xx) noexcept
            {
                return *this /= Vector_type{xx};
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator%=(const Vector_type& xx) noexcept
            {
                constexpr bool has_mod{std::is_nothrow_invocable_r_v<
                    V_, std::modulus<>, decltype(data), decltype(xx.data)>};

                if constexpr (has_mod) data %= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] %= xx.data[i];

                return *this;
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator^=(const Vector_type& xx) noexcept
            {
                constexpr bool has_xor{std::is_nothrow_invocable_r_v<
                    V_, std::bit_xor<>, decltype(data), decltype(xx.data)>};

                if constexpr (has_xor) data ^= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] ^= xx.data[i];

                return *this;
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator|=(const Vector_type& xx) noexcept
            {
                constexpr bool has_or{std::is_nothrow_invocable_r_v<
                    V_, std::bit_or<>, decltype(data), decltype(xx.data)>};

                if constexpr (has_or) data |= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] |= xx.data[i];

                return *this;
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator&=(const Vector_type& xx) noexcept
            {
                constexpr auto has_and{std::is_nothrow_invocable_r_v<
                    V_, std::bit_and<>, decltype(data), decltype(xx.data)>};

                if constexpr (has_and) data &= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] &= xx.data[i];

                return *this;
            }

            struct Rsh {
                template<typename U>
                auto operator()(const U&, const U&) const noexcept
                    -> decltype(std::declval<U>() >> std::declval<U>());
            };

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator>>=(const Vector_type& xx) noexcept
            {
                constexpr auto has_rsh{std::is_nothrow_invocable_r_v<
                    V_, Rsh, decltype(data), decltype(xx.data)>};

                if constexpr (has_rsh) data >>= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] >>= xx.data[i];

                return *this;
            }

            struct Lsh {
                template<typename U>
                auto operator()(const U&, const U&) const noexcept
                    -> decltype(std::declval<U>() << std::declval<U>());
            };

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type& operator<<=(const Vector_type& xx) noexcept
            {
                constexpr auto has_lsh{std::is_nothrow_invocable_r_v<
                    V_, Lsh, decltype(data), decltype(xx.data)>};

                if constexpr (has_lsh) data <<= xx.data;
                else for (auto i = 0u; i != rank; ++i) data[i] <<= xx.data[i];

                return *this;
            }

            // ACCESSORS
            template<typename U = T>
            __host__ __device__
            Vector_type operator+() const noexcept
            {
                return *this;
            }
            template<
                typename U = T>
                //, std::enable_if_t<std::is_signed_v<U>>* = nullptr>
            __host__ __device__
            Vector_type operator-() const noexcept
            {
                auto r{*this};
                r.data = -r.data;

                return r;
            }

            template<
                typename U = T,
                std::enable_if_t<std::is_integral_v<U>>* = nullptr>
            __host__ __device__
            Vector_type operator~() const noexcept
            {
                constexpr bool has_not{std::is_nothrow_invocable_r_v<
                    V_, std::bit_not<>, decltype(data)>};

                auto r{*this};

                if constexpr (has_not) r.data = ~r.data;
                else for (auto i = 0u; i != rank; ++i) r.data[i] = ~r.data[i];

                return r;
            }
        };
        // END STRUCT VECTOR_TYPE

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        Vector_type<T, n> operator+(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} += y;
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator+(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} += Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator+(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} += y;
        }

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        Vector_type<T, n> operator-(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} -= y;
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator-(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} -= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator-(
            U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} -= y;
        }

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        Vector_type<T, n> operator*(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} *= y;
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator*(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} *= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator*(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} *= y;
        }

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        Vector_type<T, n> operator/(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} /= y;
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator/(const Vector_type<T, n>& x, U y) noexcept
        {
            return Vector_type<T, n>{x} /= Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        Vector_type<T, n> operator/(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} /= y;
        }

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        bool operator<=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(y < x);
        }
        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        bool operator>(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return y < x;
        }
        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        bool operator >=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(x < y);
        }

        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        bool operator==(const Vector_type<T, n>& x, U y) noexcept
        {
            return x == Vector_type<T, n>{y};
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        bool operator==(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} == y;
        }

        template<typename T, std::uint32_t n>
        __host__ __device__
        inline
        bool operator!=(
            const Vector_type<T, n>& x, const Vector_type<T, n>& y) noexcept
        {
            return !(x == y);
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        bool operator!=(const Vector_type<T, n>& x, U y) noexcept
        {
            return !(x == y);
        }
        template<typename T, std::uint32_t n, typename U>
        __host__ __device__
        inline
        bool operator!=(U x, const Vector_type<T, n>& y) noexcept
        {
            return !(x == y);
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator%(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} %= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator^(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} ^= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator|(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} |= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator&(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} &= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator>>(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} >>= y;
        }

        template<
            typename T,
            std::uint32_t n,
            std::enable_if_t<std::is_integral_v<T>>* = nullptr>
        __host__ __device__
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
        __host__ __device__
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
        __host__ __device__
        inline
        Vector_type<T, n> operator<<(U x, const Vector_type<T, n>& y) noexcept
        {
            return Vector_type<T, n>{x} <<= y;
        }
    } // Namespace hip::detail.
} // Namespace hip.