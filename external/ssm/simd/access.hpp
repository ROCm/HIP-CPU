#pragma once

#include "vector.hpp"

// Types and functions for accessing individual elements within a vector.

namespace ssm
{
namespace simd
{
namespace detail
{
template <typename T, std::size_t N, std::size_t M>
struct access_impl
{
	static simd::vector<T, N> set(simd::vector<T, N> vec, T val) {
		vec[M] = val;
		return vec;
	}

	static T get(simd::vector<T, N> vec) {
		return vec[M];
	}
};
}

template <typename T, std::size_t N, std::size_t M>
inline vector<T, N> set_element(vector<T, N> vec, T val) {
	static_assert(M < N && M >= 0, "Vector access element is out of bounds");
	return detail::access_impl<T, N, M>::set(vec, val);
}

template <typename T, std::size_t N, std::size_t M>
inline T get_element(const vector<T, N> vec) {
	static_assert(M < N && M >= 0, "Vector access element is out of bounds");
	return detail::access_impl<T, N, M>::get(vec);
}

template <typename T, std::size_t N, std::size_t M>
struct accessor
{
	accessor() = default;
	accessor(T value) : vec{} {
		vec = set_element<T, N, M>(vec, value);
	}

	vector<T, N>& operator=(T value) {
		vec = set_element<T, N, M>(vec, value);
		return vec;
	}

	operator T() const {
		return get_element<T, N, M>(vec);
	}

	vector<T, N> vec;
};

template <typename T, std::size_t N>
struct vector_data
{
	vector<T, N> data = {};
};

template <typename T>
struct vector_data<T, 2>
{
	union {
		accessor<T, 2, 0> x;
		accessor<T, 2, 1> y;
		vector<T, 2> data = {};
	};
};

template <typename T>
struct vector_data<T, 3>
{
	union {
		simd::accessor<T, 3, 0> x;
		simd::accessor<T, 3, 1> y;
		simd::accessor<T, 3, 2> z;
		simd::vector<T, 3> data = {};
	};
};

template <typename T>
struct vector_data<T, 4>
{
	union {
		simd::accessor<T, 4, 0> x;
		simd::accessor<T, 4, 1> y;
		simd::accessor<T, 4, 2> z;
		simd::accessor<T, 4, 3> w;
		simd::vector<T, 4> data = {};
	};
};
}
}
