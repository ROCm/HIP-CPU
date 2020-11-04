#pragma once

#include "detail/setup.hpp"
#include "detail/vector-manip.hpp"

namespace ssm {
template <typename T, std::size_t N>
struct vector : generic_vec<T, N> {
  using generic_vec<T, N>::generic_vec;
};

template <typename T, std::size_t N>
inline T dot(const vector<T, N>& a, const vector<T, N>& b) {
  return detail::vec_impl<T, N>::dot(a, b);
}

template <typename T, std::size_t N>
inline T sqlength(const vector<T, N>& vec) {
  return dot(vec, vec);
}

template <typename T, std::size_t N>
inline T length(const vector<T, N>& vec) {
  return static_cast<T>(std::sqrt(sqlength(vec)));
}

template <typename T, std::size_t N>
inline T distance(const vector<T, N>& a, const vector<T, N>& b) {
  return length(a - b);
}

template <typename T, std::size_t N>
inline vector<T, N> cross(const vector<T, N>& a, const vector<T, N>& b) {
  static_assert(N == 3, "Cross product is only defined for 3D vectors");
  return vector<T, N>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

template <typename T, std::size_t N>
inline vector<T, N> normalize(vector<T, N> vec) {
  detail::vec_impl<T, N>::normalize(vec);
  return vec;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator+=(vector<T, N>& a, const vector<T, N>& b) {
  detail::vec_impl<T, N>::add(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator-=(vector<T, N>& a, const vector<T, N>& b) {
  detail::vec_impl<T, N>::sub(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator*=(vector<T, N>& a, const vector<T, N>& b) {
  detail::vec_impl<T, N>::mul(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator/=(vector<T, N>& a, const vector<T, N>& b) {
  detail::vec_impl<T, N>::div(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator*=(vector<T, N>& a, T b) {
  detail::vec_impl<T, N>::mul(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator/=(vector<T, N>& a, T b) {
  detail::vec_impl<T, N>::div(a, b);
  return a;
}

template <typename T, std::size_t N>
inline vector<T, N> operator+(vector<T, N> a, const vector<T, N>& b) {
  return a += b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator-(vector<T, N> a, const vector<T, N>& b) {
  return a -= b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator*(vector<T, N> a, const vector<T, N>& b) {
  return a *= b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator*(vector<T, N> a, T b) {
  return a *= b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator*(T a, const vector<T, N>& b) {
  return b * a;
}

template <typename T, std::size_t N>
inline vector<T, N> operator/(vector<T, N> a, const vector<T, N>& b) {
  return a /= b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator/(vector<T, N> a, T b) {
  return a /= b;
}

template <typename T, std::size_t N>
inline vector<T, N> operator-(vector<T, N> vec) {
  detail::vec_impl<T, N>::negate(vec);
  return vec;
}

template <typename T, std::size_t N>
inline bool operator==(const vector<T, N>& a, const vector<T, N>& b) {
  return detail::vec_impl<T, N>::equals(a, b);
}

template <typename T, std::size_t N>
inline bool operator!=(const vector<T, N>& a, const vector<T, N>& b) {
  return !(a == b);
}

//----------------------------------------------
// Utility typedefs
//----------------------------------------------
template <std::size_t N>
using vec = vector<float, N>;
template <std::size_t N>
using dvec = vector<double, N>;
template <std::size_t N>
using ivec = vector<int, N>;
template <std::size_t N>
using uvec = vector<unsigned, N>;

using vec2 = vec<2>;
using vec3 = vec<3>;
using vec4 = vec<4>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;
using uvec2 = uvec<2>;
using uvec3 = uvec<3>;
using uvec4 = uvec<4>;
} // namespace ssm
