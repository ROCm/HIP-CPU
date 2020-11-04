#pragma once

#include "detail/loop-unroll.hpp"
#include "vector.hpp"

namespace ssm {
template <typename T, std::size_t M, std::size_t N = M>
class matrix {
public:
  using column_type = vector<T, M>;
  static constexpr std::size_t height = M;
  static constexpr std::size_t width = N;

  matrix() = default;

  explicit matrix(T val) {
    for (std::size_t i = 0; i < N; ++i)
      data[i] = vector<T, M>(val);
  }

  column_type& operator[](std::size_t index) {
    assert(index < N);
    return data[index];
  }
  const column_type& operator[](std::size_t index) const {
    assert(index < N);
    return data[index];
  }

  template <std::size_t I>
  column_type get() const {
    return data[I];
  }
  template <std::size_t I>
  void set(const column_type& vec) {
    data[I] = vec;
  }

  T* begin() { return data[0].begin(); }
  const T* begin() const { return data[0].begin(); }
  T* end() { return data[N - 1].end(); }
  const T* end() const { return data[N - 1].end(); }

private:
  std::array<column_type, N> data = {};
};

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator-(matrix<T, M, N> mat) {
  for (std::size_t i = 0; i < N; ++i)
    mat[i] = -mat[i];
  return mat;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N>& operator+=(matrix<T, M, N>& a,
                                   const matrix<T, M, N>& b) {
  for (std::size_t i = 0; i < N; ++i)
    a[i] += b[i];
  return a;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N>& operator-=(matrix<T, M, N>& a,
                                   const matrix<T, M, N>& b) {
  for (std::size_t i = 0; i < N; ++i)
    a[i] -= b[i];
  return a;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator+(matrix<T, M, N> a, const matrix<T, M, N>& b) {
  return a += b;
}
template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator-(matrix<T, M, N> a, const matrix<T, M, N>& b) {
  return a -= b;
}

template <typename T, std::size_t M, std::size_t N>
inline bool operator==(const matrix<T, M, N>& a, const matrix<T, M, N>& b) {
  for (std::size_t i = 0; i < N; ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

template <typename T, std::size_t M, std::size_t N>
inline bool operator!=(const matrix<T, M, N>& a, const matrix<T, M, N>& b) {
  return !(a == b);
}

template <typename T, std::size_t M, std::size_t N>
inline vector<T, M> operator*(const vector<T, N>& vec,
                              const matrix<T, M, N>& mat) {
  return detail::unroll<0, N>::inner_product(vec, mat);
}

template <typename T, std::size_t M, std::size_t N>
inline vector<T, M> operator*(const matrix<T, M, N>& mat,
                              const vector<T, N>& vec) {
  return vec * mat;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N>& operator*=(matrix<T, M, N>& mat, T val) {
  for (std::size_t i = 0; i < N; ++i)
    mat[i] *= val;
  return mat;
}

template <typename T, std::size_t N>
inline vector<T, N>& operator*=(vector<T, N>& vec, const matrix<T, N, N>& mat) {
  vec = detail::unroll<0, N>::inner_product(vec, mat);
  return vec;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator*(matrix<T, M, N> mat, T val) {
  return mat *= val;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator*(T val, matrix<T, M, N> mat) {
  return mat *= val;
}

template <typename T, std::size_t M, std::size_t N, std::size_t P>
inline matrix<T, M, P> operator*(const matrix<T, M, N>& a,
                                 const matrix<T, N, P> b) {
  matrix<T, M, P> ret;
  for (std::size_t i = 0; i < M; ++i)
    ret[i] = b[i] * a;
  return ret;
}

template <typename T, std::size_t N>
inline matrix<T, N, N>& operator*=(matrix<T, N, N>& a,
                                   const matrix<T, N, N> b) {
  a = a * b;
  return a;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N>& operator/=(matrix<T, M, N>& mat, T val) {
  for (std::size_t i = 0; i < N; ++i)
    mat[i] /= val;
  return mat;
}

template <typename T, std::size_t M, std::size_t N>
inline matrix<T, M, N> operator/(matrix<T, M, N> mat, T val) {
  return mat /= val;
}

template <typename T, std::size_t M, std::size_t N>
inline T* data_ptr(matrix<T, M, N>& mat) {
  return data_ptr(mat[0]);
}

template <typename T, std::size_t M, std::size_t N>
inline const T* data_ptr(const matrix<T, M, N>& mat) {
  return data_ptr(mat[0]);
}

//----------------------------------------------
// Utility typedefs
//----------------------------------------------
using mat2 = matrix<float, 2, 2>;
using mat3 = matrix<float, 3, 3>;
using mat4 = matrix<float, 4, 4>;
using dmat2 = matrix<double, 2, 2>;
using dmat3 = matrix<double, 3, 3>;
using dmat4 = matrix<double, 4, 4>;
using imat2 = matrix<int, 2, 2>;
using imat3 = matrix<int, 3, 3>;
using imat4 = matrix<int, 4, 4>;
using umat2 = matrix<unsigned, 2, 2>;
using umat3 = matrix<unsigned, 3, 3>;
using umat4 = matrix<unsigned, 4, 4>;
} // namespace ssm
