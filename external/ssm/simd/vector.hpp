#pragma once

#include <array>
#include <cstddef>

namespace ssm
{
namespace simd
{
template <typename T, std::size_t N>
struct make_vector
{
	using type = std::array<T, N>;
};

template <typename T, std::size_t N>
using vector = typename make_vector<T, N>::type;

template <typename Arr, typename T>
inline void fill(Arr& vec, T val) {
	vec.fill(val);
}

template <typename Arr, std::size_t>
inline void assign(Arr& vec) {}

template <typename Arr, std::size_t I = 0, typename U, typename... Args>
inline void assign(Arr& vec, U val, Args... args) {
	vec[I] = val;
	assign<Arr, I + 1, Args...>(vec, args...);
}
}
}
