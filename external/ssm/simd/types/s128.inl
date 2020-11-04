#if SSM_ARCH & SSM_ARCH_SSE2_BIT
namespace ssm
{
namespace simd
{
template<>
struct is_simd<int, 4> { static constexpr bool value = true; };

typedef __m128i s128;

namespace detail
{
template <size_t N>
struct access_impl<int, 4, N>
{
	static inline s128 set(s128 vec, int val) {
#if SSM_ARCH & SSM_ARCH_SSE4_BIT
		return _mm_insert_epi32(vec, val, N);
#else
		const s128 ins0 = _mm_insert_epi16(vec, val, N / 2);
		return _mm_insert_epi16(ins0, val >> 16, N / 2 + 1);
#endif
	}

	static inline int get(s128 vec) {
#if SSM_ARCH & SSM_ARCH_SSE4_BIT
		return _mm_extract_epi32(vec, N);
#else
		const s128 shuf = _mm_shuffle_epi32(vec, _MM_SHUFFLE(N, N, N, N));
		return _mm_cvtsi128_si32(shuf);
#endif
	}
};

template <>
struct access_impl<int, 4, 0>
{
	static inline s128 set(s128 vec, int val) {
		const s128 ins0 = _mm_insert_epi16(vec, val, 0);
		return _mm_insert_epi16(ins0, val >> 16, 1);
	}

	static inline int get(s128 vec) {
		return _mm_cvtsi128_si32(vec);
	}
};
}

template <>
struct make_vector<int, 4>
{
	using type = s128;
};

inline void assign(s128& vec, int a, int b, int c, int d) {
	vec = _mm_set_epi32(d, c, b, a);
}

inline void fill(s128& vec, int a) {
	vec =  _mm_set1_epi32(a);
}

inline s128 add(s128 a, s128 b) {
	return _mm_add_epi32(a, b);
}

inline s128 sub(s128 a, s128 b) {
	return _mm_sub_epi32(a, b);
}

inline s128 mul(s128 a, s128 b) {
#if SSM_ARCH & SSM_ARCH_SSE4_BIT
    return _mm_mullo_epi32(a, b);
#else
    const s128 tmp1 = _mm_mul_epu32(a, b);
    const s128 tmp2 = _mm_mul_epu32(_mm_srli_si128(a,4), _mm_srli_si128(b,4));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0)));
#endif
}

inline s128 div(s128 a, s128 b) {
	// Convert to float, divide, convert back
	const __m128 div = _mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b));
	return _mm_cvttps_epi32(div);
}

// Helpers for creating compile-time vectors
// used for negation.
#if SSM_ARCH & SSM_ARCH_SSSE3_BIT
static constexpr int sign_mask(bool a) {
	return a ? -1 : 1;
}

template <bool X, bool Y, bool Z, bool W>
inline s128 negate(s128 vec) {
	const s128 sign = _mm_set_epi32(sign_mask(W), sign_mask(Z), sign_mask(Y), sign_mask(X));
	return _mm_sign_epi32(vec, sign);
}
#else
// AND this values with the vector to produce a mask,
// so 0xFF keeps the same value there was before.
static constexpr int sign_mask(bool a) {
	return a ? 0 : 0xFFFFFFFF;
}

template <bool X, bool Y, bool Z, bool W>
inline s128 negate(s128 vec) {
	const s128 vec2 = _mm_add_epi32(vec, vec);
	const s128 mask = _mm_set_epi32(sign_mask(W), sign_mask(Z), sign_mask(Y), sign_mask(X));
	const s128 and0 = _mm_and_si128(vec2, mask);
	return _mm_sub_epi32(and0, vec);
}
#endif

inline s128 negate(s128 vec) {
	const s128 zero = _mm_setzero_si128();
	return _mm_sub_epi32(zero, vec);
}

template <size_t N>
inline s128 shuffle(s128 a) {
	static_assert(N < 4, "Shuffle index out of range");
	return _mm_shuffle_epi32(a, (N << 6) | (N << 4) | (N << 2) | N);
}

template <size_t X, size_t Y, size_t Z, size_t W>
inline s128 shuffle(s128 a) {
	static_assert(X < 4, "Shuffle index out of range");
	static_assert(Y < 4, "Shuffle index out of range");
	static_assert(Z < 4, "Shuffle index out of range");
	static_assert(W < 4, "Shuffle index out of range");
	return _mm_shuffle_epi32(a, (W << 6) | (Z << 4) | (Y << 2) | X);
}

inline bool equals(s128 a, s128 b) {
	const s128 cmp0 = _mm_cmpeq_epi32(a, b);
	return _mm_movemask_epi8(cmp0) == 0xffff;
}

inline s128 dot(s128 a, s128 b) {
#if SSM_ARCH & SSM_ARCH_SSSE3_BIT
	const s128 mul1 = mul(a, b);
	const s128 hadd = _mm_hadd_epi32(mul1, mul1);
	return _mm_hadd_epi32(hadd, hadd);
#else
	const s128 mul0 = mul(a, b);
	const s128 swp0 = shuffle<2, 3, 0, 1>(mul0);
	const s128 add0 = add(mul0, swp0);
	const s128 swp1 = shuffle<3, 2, 1, 0>(add0);
	return add(add0, swp1);
#endif
}
}
}
#endif
