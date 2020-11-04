#if SSM_ARCH & SSM_ARCH_SSE2_BIT
namespace ssm
{
namespace simd
{
template<>
struct is_simd<float, 4> { static constexpr bool value = true; };

typedef __m128 f128;

namespace detail
{
template <size_t N>
struct access_impl<float, 4, N>
{
	static inline f128 set(f128 vec, float val) {
#if SSM_ARCH & SSM_ARCH_SSE4_BIT
		return _mm_insert_ps(vec, _mm_set_ss(val), N * 0x10);
#else
		// Swap first and Nth element
		const f128 shuf1 = _mm_shuffle_ps(vec, vec, (0xE4 | N) & ~(3 << (2 * N)));
		// Move new element in first position
		const f128 single = _mm_set_ss(val);
		// Swap again
		const f128 mov = _mm_move_ss(shuf1, single);
		return _mm_shuffle_ps(mov, mov, (0xE4 | N) & ~(3 << (2 * N)));
#endif
	}

	static inline float get(f128 vec) {
		const f128 shuf = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(N, N, N, N));
		return _mm_cvtss_f32(shuf);
	}
};

template <>
struct access_impl<float, 4, 0>
{
	static inline f128 set(f128 vec, float val) {
		const f128 single = _mm_set_ss(val);
		return _mm_move_ss(vec, single);
	}

	static inline float get(f128 vec) {
		return _mm_cvtss_f32(vec);
	}
};
}

template <>
struct make_vector<float, 4>
{
	using type = f128;
};

inline void assign(f128& vec, float a, float b, float c, float d) {
	vec = _mm_set_ps(d, c, b, a);
}

inline void fill(f128& vec, float a) {
	vec = _mm_set1_ps(a);
}

inline f128 add(f128 a, f128 b) {
	return _mm_add_ps(a, b);
}

inline f128 sub(f128 a, f128 b) {
	return _mm_sub_ps(a, b);
}

inline f128 mul(f128 a, f128 b) {
	return _mm_mul_ps(a, b);
}

inline f128 div(f128 a, f128 b) {
	return _mm_div_ps(a, b);
}

template <bool X, bool Y, bool Z, bool W>
inline f128 negate(f128 vec) {
	const f128 sign = _mm_set_ss(-0.f);
	const f128 shuf = _mm_shuffle_ps(sign, sign, (!W << 6) | (!Z << 4) | (!Y << 2) | !X);
	return _mm_xor_ps(vec, shuf);
}

inline f128 negate(f128 vec) {
	const f128 sign = _mm_set1_ps(-0.f);
	return _mm_xor_ps(vec, sign);
}

template <size_t N>
inline f128 shuffle(f128 a) {
	static_assert(N < 4, "Shuffle index out of range");
	return _mm_shuffle_ps(a, a, (N << 6) | (N << 4) | (N << 2) | N);
}

template <size_t X, size_t Y, size_t Z, size_t W>
inline f128 shuffle(f128 a, f128 b) {
	static_assert(X < 4, "Shuffle index out of range");
	static_assert(Y < 4, "Shuffle index out of range");
	static_assert(Z < 4, "Shuffle index out of range");
	static_assert(W < 4, "Shuffle index out of range");
	return _mm_shuffle_ps(a, b, (W << 6) | (Z << 4) | (Y << 2) | X);
}

inline bool equals(f128 a, f128 b) {
	const f128 cmp0 = _mm_cmpeq_ps(a, b);
	return _mm_movemask_ps(cmp0) == 0xf;
}

inline f128 rsqrt(f128 a) {
#ifdef SSM_FAST_MATH
	return _mm_rsqrt_ps(a);
#else
	const f128 three = _mm_set1_ps(3.0f);
	const f128 half = _mm_set1_ps(0.5f);
	const f128 res = _mm_rsqrt_ps(a); 
	const f128 muls = _mm_mul_ps(_mm_mul_ps(a, res), res); 
	return _mm_mul_ps(_mm_mul_ps(half, res), _mm_sub_ps(three, muls)); 
#endif
}

inline f128 dot(f128 a, f128 b) {
#if SSM_ARCH & SSM_ARCH_SSE4_2_BIT
	return _mm_dp_ps(a, b, 0xFF);
#elif SSM_ARCH & SSM_ARCH_SSE3_BIT
	const f128 mul1 = mul(a, b);
	const f128 hadd = _mm_hadd_ps(mul1, mul1)
	return _mm_hadd_ps(hadd, hadd)
#else
	const f128 mul0 = mul(a, b);
	const f128 swp0 = shuffle<2, 3, 0, 1>(mul0, mul0);
	const f128 add0 = add(mul0, swp0);
	const f128 swp1 = shuffle<3, 2, 1, 0>(add0, add0);
	return add(add0, swp1);
#endif
}
}
}
#endif
