#pragma once

// The following are types and functions for achieving "compile-time loops",
// using simple SFINAE template metaprogramming.

#include <cstdint>

namespace ssm {
namespace detail {
// returns a if A and B are equal, b otherwise.
template <std::size_t A, std::size_t B, typename T>
constexpr T select(T a, T b) {
  return A == B ? a : b;
}

template <std::size_t Start, std::size_t End>
struct unroll {
  static_assert(
      Start < End,
      "Error in template parameters: \"Start\" must be less than \"End\"");
  using next = unroll<Start + 1, End>;

  template <typename Vec, typename Mat>
  inline static typename Mat::column_type inner_product(Vec& vec,
                                                        const Mat& mat) {
    return mat[Start] * vec.template get<Start>() +
           next::inner_product(vec, mat);
  }

  // Generates the I'th vector of an identity matrix.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec identity_vec(Args... args) {
    return next::template identity_vec<Vec, I, typename Vec::value_type,
                                       Args...>(
        args..., select<Start, I, typename Vec::value_type>(1, 0));
  }

  // Generates the I'th vector of a scaling matrix.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec scaling_vec(const Vec& vec, Args... args) {
    return next::template scaling_vec<Vec, I, typename Vec::value_type,
                                      Args...>(
        vec, args...,
        select<Start, I, typename Vec::value_type>(vec.template get<Start>(),
                                                   0));
  }
  // Generates a vector that scales a matrix by vec.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec scaling_inplace_vec(const Vec& vec, Args... args) {
    return next::template scaling_vec<Vec, I, typename Vec::value_type,
                                      Args...>(
        vec, args...,
        select<Start, I, typename Vec::value_type>(vec.template get<Start>(),
                                                   1));
  }

  template <typename H, typename NH, typename... Args>
  inline static NH dehomogenize_vec(const H& vec, Args... args) {
    return next::template dehomogenize_vec<H, NH, typename H::value_type,
                                           Args...>(vec, args...,
                                                    vec.template get<Start>());
  }

  template <typename H, typename NH, typename... Args>
  inline static H extend_vec(const NH& vec, Args... args) {
    return next::template extend_vec<H, NH, typename NH::value_type, Args...>(
        vec, vec.template get<NH::size - Start - 1>(), args...);
  }

  template <typename Vec, typename NewVec, typename... Args>
  inline static NewVec convert_vec(const Vec& vec, Args... args) {
    using NewT = typename NewVec::value_type;
    return next::template convert_vec<Vec, NewVec, Args..., NewT>(
        vec, args..., static_cast<NewT>(vec.template get<Start>()));
  }

  template <typename Mat>
  inline static void identity_mat(Mat& out_mat) {
    out_mat[Start] =
        unroll<0, End>::template identity_vec<typename Mat::column_type,
                                              Start>();
    next::identity_mat(out_mat);
  }

  template <typename Mat, typename Vec>
  inline static void scaling_mat(Mat& out_mat, const Vec& vec) {
    out_mat[Start] = unroll<0, End>::template scaling_vec<Vec, Start>(vec);
    next::scaling_mat(out_mat, vec);
  }
  template <typename Mat, typename Vec>
  inline static void scale_inplace_mat(Mat& mat, const Vec& vec) {
    mat[Start] *= unroll<0, End>::template scaling_inplace_vec<Vec, Start>(vec);
    next::scale_inplace_mat(mat, vec);
  }
};

template <std::size_t End>
struct unroll<End, End> {
  template <typename Vec, typename Mat>
  static inline typename Mat::column_type inner_product(Vec& vec,
                                                        const Mat& mat) {
    return {};
  }

  // Generates the I'th vector of an identity matrix.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec identity_vec(Args... args) {
    return Vec(args...);
  }

  // Generates the I'th vector of a scaling matrix.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec scaling_vec(const Vec& vec, Args... args) {
    return Vec(args...);
  }
  // Generates the I'th vector of a scaling matrix.
  template <typename Vec, std::size_t I, typename... Args>
  inline static Vec scaling_inplace_vec(const Vec& vec, Args... args) {
    return Vec(args...);
  }

  template <typename H, typename NH, typename... Args>
  inline static H extend_vec(const NH& vec, Args... args) {
    return H(args...);
  }

  template <typename H, typename NH, typename... Args>
  inline static NH dehomogenize_vec(const H& vec, Args... args) {
    return NH(args...);
  }

  template <typename Vec, typename NewVec, typename... Args>
  inline static NewVec convert_vec(const Vec& vec, Args... args) {
    return NewVec(args...);
  }

  template <typename Mat>
  inline static void identity_mat(Mat& out_mat) {}

  template <typename Mat, typename Vec>
  inline static void scaling_mat(Mat& out_mat, const Vec& vec) {}

  template <typename Mat, typename Vec>
  inline static void scale_inplace_mat(Mat& out_mat, const Vec& vec) {}
};
} // namespace detail
} // namespace ssm
