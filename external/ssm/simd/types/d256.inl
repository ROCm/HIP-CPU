#if SSM_ARCH & SSM_ARCH_AVX_BIT
namespace ssm
{
namespace simd
{
template<>
struct is_simd<double, 4> { static constexpr bool value = true; };

typedef __m256d d256;

namespace detail
{
template <size_t N>
struct access_impl<double, 4, N>
{
	static inline d256 set(d256 vec, double val) {
		const d128 extr = _mm256_extractf128_pd(vec, N / 2);
		const d128 set0 = access_impl<double, 2, N % 2>::set(temp, val);
		return _mm256_insertf128_pd(vec, set0, N / 2);
	}

	static inline double get(d256 vec) {
		const d128 extr = _mm256_extractf128_pd(vec, N / 2);
		return access_impl<double, 2, N % 2>::get(extr);
	}
}
}

template <>
struct make_vector<double, 4>
{
	using type = d256;
};

inline void assign(d256& vec, double a, double b, double c, double d) {
	vec = _mm256_set_pd(d, c, b, a);
}

inline void fill(d256& vec, double a) {
	vec = _mm256_set1_pd(a);
}

inline d256 add(d256 a, d256 b) {
	return _mm256_add_pd(a, b);
}

inline d256 sub(d256 a, d256 b) {
	return _mm256_sub_pd(a, b);
}

inline d256 mul(d256 a, d256 b) {
	return _mm256_mul_pd(a, b);
}

inline d256 div(d256 a, d256 b) {
	return _mm256_div_pd(a, b);
}

template <bool X, bool Y, bool Z, bool W>
inline d256 negate(d256 vec) {
	const d256 sign = _mm256_set_ss(-0);
	const d256 shuf = _mm256_shuffle_pd(sign, sign, (!W << 6) | (!Z << 4) | (!Y << 2) | !X);
	return _mm256_xor_pd(vec, shuf);
}

inline d256 negate(d256 vec) {
	const d256 sign = _mm256_set1_pd(-0.f);
	return _mm256_xor_pd(vec, sign);
}

template <size_t N>
inline d256 shuffle(d256 a) {
	static_assert(N < 4, "Shuffle index out of range");
	return _mm256_shuffle_pd(a, a, _MM_SHUFFLE(N, N, N, N));
}

template <size_t X, size_t Y, size_t Z, size_t W>
inline d256 shuffle(d256 a, d256 b) {
	static_assert(X < 4, "Shuffle index out of range");
	static_assert(Y < 4, "Shuffle index out of range");
	static_assert(Z < 4, "Shuffle index out of range");
	static_assert(W < 4, "Shuffle index out of range");
	return _mm256_shuffle_pd(a, b, (W << 6) | (Z << 4) | (Y << 2) | X);
}

inline bool equals(d256 a, d256 b) {
	const d256 cmp0 = _mm256_cmpeq_pd(a, b);
	return _mm256_movemask_pd(cmp0) == 0xf;
}

inline d256 rsqrt(d256 a) {
#ifdef SSM_FAST_MATH
	return _mm256_rsqrt_pd(a);
#else
	const d256 three = _mm256_set1_pd(3.0f);
	const d256 half = _mm256_set1_pd(0.5f);
	const d256 res = _mm256_rsqrt_pd(a); 
	const d256 muls = _mm256_mul_pd(_mm256_mul_pd(a, res), res); 
	return _mm256_mul_pd(_mm256_mul_pd(half, res), _mm256_sub_pd(three, muls)); 
#endif
}

inline d256 dot(d256 a, d256 b) {
#if SSM_ARCH & SSM_ARCH_SSE4_2_BIT
	return _mm256_dp_pd(a, b, 0xFF);
#elif SSM_ARCH & SSM_ARCH_SSE3_BIT
	const d256 mul1 = mul(a, b);
	const d256 hadd = _mm256_hadd_pd(mul1, mul1)
	return _mm256_hadd_pd(hadd, hadd)
#else
	const d256 mul0 = mul(a, b);
	const d256 swp0 = shuffle<2, 3, 0, 1>(mul0, mul0);
	const d256 add0 = add(mul0, swp0);
	const d256 swp1 = shuffle<3, 2, 1, 0>(add0, add0);
	return add(add0, swp1);
#endif
}
}
}
#endif
