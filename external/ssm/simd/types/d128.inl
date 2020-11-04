#if SSM_ARCH & SSM_ARCH_SSE2_BIT
namespace ssm
{
namespace simd
{
template<>
struct is_simd<double, 2> { static constexpr bool value = true; };

typedef __m128d d128;

namespace detail
{
template <>
struct access_impl<double, 2, 1>
{
	static inline d128 set(d128 vec, double val) {
		const d128 wide = _mm_set1_pd(val);
		return _mm_move_sd(wide, vec);
	}

	static inline double get(d128 vec) {
		const d128 shuf = _mm_shuffle_pd(vec, vec, 3);
		return _mm_cvtsd_f64(shuf);
	}
};
template <>
struct access_impl<double, 2, 0>
{
	static inline d128 set(d128 vec, double val) {
		const d128 singl = _mm_set_sd(val);
		return _mm_move_sd(vec, singl);
	}

	static inline double get(d128 vec) {
		return _mm_cvtsd_f64(vec);
	}
};
}

template <>
struct make_vector<double, 2>
{
	using type = d128;
};

inline void assign(d128& vec, double a, double b) {
	vec = _mm_set_pd(b, a);
}

inline void fill(d128& vec, double a) {
	vec = _mm_set1_pd(a);
}

inline d128 add(d128 a, d128 b) {
	return _mm_add_pd(a, b);
}

inline d128 sub(d128 a, d128 b) {
	return _mm_sub_pd(a, b);
}

inline d128 mul(d128 a, d128 b) {
	return _mm_mul_pd(a, b);
}

inline d128 div(d128 a, d128 b) {
	return _mm_div_pd(a, b);
}

template <bool X, bool Y>
inline d128 negate(d128 vec) {
	const d128 sign = _mm_set_pd(X ? 0. : -0., Y ? 0. : -0.);
	return _mm_xor_pd(vec, sign);
}

inline d128 negate(d128 vec) {
	const d128 sign = _mm_set1_pd(-0.f);
	return _mm_xor_pd(vec, sign);
}

template <size_t N>
inline d128 shuffle(d128 a) {
	static_assert(N < 2, "Shuffle index out of range");
	return _mm_shuffle_pd(a, a, (N << 1) | N);
}

template <size_t X, size_t Y>
inline d128 shuffle(d128 a, d128 b) {
	static_assert(X < 2, "Shuffle index out of range");
	static_assert(Y < 2, "Shuffle index out of range");
	return _mm_shuffle_pd(a, b, (Y << 1) | X);
}

inline bool equals(d128 a, d128 b) {
	const d128 cmp0 = _mm_cmpeq_pd(a, b);
	return _mm_movemask_pd(cmp0) == 0x3;
}

inline d128 rsqrt(d128 a) {
	const d128 sqrt = _mm_sqrt_pd(a);
	return _mm_div_pd(_mm_set1_pd(1.0f), sqrt);
}

inline d128 dot(d128 a, d128 b) {
#if SSM_ARCH & SSM_ARCH_SSE4_2_BIT
	return _mm_dp_pd(a, b, 0xFF);
#elif SSM_ARCH & SSM_ARCH_SSE3_BIT
	const d128 mul1 = mul(a, b);
	return _mm_hadd_pd(mul1, mul1)
#else
	const d128 mul0 = mul(a, b);
	const d128 swp0 = shuffle<1, 0>(mul0, mul0);
	return add(mul0, swp0);
#endif
}
}
}
#endif
