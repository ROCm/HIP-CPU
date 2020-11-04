#pragma once

#include "detail/setup.hpp"
#include "detail/vector-manip.hpp"

namespace ssm {
template <typename T>
struct quaternion : generic_vec<T, 4> {
  using generic_vec<T, 4>::generic_vec;
};

template <typename T>
inline quaternion<T> euler(T yaw, T pitch, T roll) {
  T cy = cos(yaw * 0.5);
  T sy = sin(yaw * 0.5);
  T cp = cos(pitch * 0.5);
  T sp = sin(pitch * 0.5);
  T cr = cos(roll * 0.5);
  T sr = sin(roll * 0.5);

  T w = cy * cp * cr + sy * sp * sr;
  T x = cy * cp * sr - sy * sp * cr;
  T y = sy * cp * sr + cy * sp * cr;
  T z = sy * cp * cr - cy * sp * sr;

  return quaternion<T>(x, y, z, w);
}

template <typename T>
inline T dot(const quaternion<T>& a, const quaternion<T>& b) {
  return detail::vec_impl<T, 4>::dot(a, b);
}

template <typename T>
inline T sqnorm(const quaternion<T>& q) {
  return dot(q, q);
}

template <typename T>
inline T norm(const quaternion<T>& q) {
  return static_cast<T>(std::sqrt(sqnorm(q)));
}

template <typename T>
inline quaternion<T> conjugate(quaternion<T> q) {
  detail::vec_impl<T, 4>::quat_conjugate(q);
  return q;
}

template <typename T>
inline quaternion<T> inverse(quaternion<T> q) {
  detail::vec_impl<T, 4>::quat_conjugate(q);
  return q / sqnorm(q);
}

// Specific form for unit quaternions, since it's much simpler
/*
template <typename T>
inline unit_quaternion<T> inverse(unit_quaternion<T> q) {
  return conjugate(q);
}
*/

template <typename T>
inline quaternion<T> normalize(quaternion<T> q) {
  detail::vec_impl<T, 4>::normalize(q);
  return q;
}

template <typename T>
inline quaternion<T>& operator+=(quaternion<T>& a, const quaternion<T>& b) {
  detail::vec_impl<T, 4>::add(a, b);
  return a;
}

template <typename T>
inline quaternion<T>& operator-=(quaternion<T>& a, const quaternion<T>& b) {
  detail::vec_impl<T, 4>::sub(a, b);
  return a;
}

template <typename T>
inline quaternion<T>& operator*=(quaternion<T>& a, const quaternion<T>& b) {
  detail::vec_impl<T, 4>::quat_mul(a, b);
  return a;
}

template <typename T>
inline quaternion<T>& operator*=(quaternion<T>& a, T b) {
  detail::vec_impl<T, 4>::mul(a, b);
  return a;
}

template <typename T>
inline quaternion<T>& operator/=(quaternion<T>& a, T b) {
  detail::vec_impl<T, 4>::div(a, b);
  return a;
}

template <typename T>
inline quaternion<T> operator+(quaternion<T> a, const quaternion<T>& b) {
  return a += b;
}

template <typename T>
inline quaternion<T> operator-(quaternion<T> a, const quaternion<T>& b) {
  return a -= b;
}

template <typename T>
inline quaternion<T> operator*(quaternion<T> a, const quaternion<T>& b) {
  return a *= b;
}

template <typename T>
inline quaternion<T> operator*(quaternion<T> a, T b) {
  return a *= b;
}

template <typename T>
inline quaternion<T> operator*(T a, const quaternion<T>& b) {
  return b * a;
}

template <typename T>
inline quaternion<T> operator/(quaternion<T> a, T b) {
  return a /= b;
}

template <typename T>
inline quaternion<T> operator-(quaternion<T> q) {
  detail::vec_impl<T, 4>::negate(q);
  return q;
}

template <typename T>
inline bool operator==(const quaternion<T>& a, const quaternion<T>& b) {
  return detail::vec_impl<T, 4>::equals(a, b);
}

template <typename T>
inline bool operator!=(const quaternion<T>& a, const quaternion<T>& b) {
  return !(a == b);
}

//----------------------------------------------
// Utility typedefs
//----------------------------------------------
using quat = quaternion<float>;
using dquat = quaternion<double>;
} // namespace ssm
