#pragma once

#include <type_traits>

namespace ssm
{
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T>
using elem_type = typename std::remove_reference<
	typename std::remove_cv<decltype((std::declval<T>())[0])>::type
>::type;

namespace simd
{
// value is true if the specified vector type can be accelerated using SIMD,
// for example is_simd<float, 4> or is_simd<double, 2>
template<typename T, int N>
struct is_simd { static constexpr bool value = false; };
}

template <typename T, typename... Args>
using fst_t = T;
}
