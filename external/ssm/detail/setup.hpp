#pragma once

//Platform
#ifdef SSM_FORCE_PLATFORM_UNKNOWN
#define SSM_PLATFORM_UNKNOWN
#elif defined(_WIN32)
#define SSM_PLATFORM_WINDOWS
#elif defined(__ANDROID__)
#define SSM_PLATFORM_ANDROID
#elif defined(__linux)
#define SSM_PLATFORM_LINUX
#elif defined(__APPLE__)
#define SSM_PLATFORM_APPLE
#elif defined(__unix)
#define SSM_PLATFORM_UNIX
#elif defined(__native_client__)
#define SSM_PLATFORM_NATIVE
#else
#define SSM_PLATFORM_UNKNOWN
#endif

//Compilers
#ifdef SSM_FORCE_COMPILER_UNKNOWN
#define SSM_COMPILER_UNKNOWN
#elif defined(__INTEL_COMPILER)
#define SSM_COMPILER_INTEL
#elif defined(__clang)
#define SSM_COMPILER_CLANG
#elif defined(_MSC_VER)
#define SSM_COMPILER_VC
#elif defined(__GNUC__) || defined(__MINGW32__)
#define SSM_COMPILER_GCC
#else
#define SSM_COMPILER_UNKNOWN
#endif

//Processor architectures
#define SSM_ARCH_PURE 0
#define SSM_ARCH_SSE2 1
#define SSM_ARCH_SSE3 3
#define SSM_ARCH_SSSE3 7
#define SSM_ARCH_SSE4 15
#define SSM_ARCH_SSE4_2 31
#define SSM_ARCH_AVX 63
#define SSM_ARCH_AVX2 127
#define SSM_ARCH_AVX512 255

#define SSM_ARCH_SSE2_BIT 1
#define SSM_ARCH_SSE3_BIT 2
#define SSM_ARCH_SSSE3_BIT 4
#define SSM_ARCH_SSE4_BIT 8
#define SSM_ARCH_SSE4_2_BIT 16
#define SSM_ARCH_AVX_BIT 32
#define SSM_ARCH_AVX2_BIT 64
#define SSM_ARCH_AVX512_BIT 128

#ifdef SSM_FORCE_PURE
# define SSM_ARCH SSM_ARCH_PURE
#elif defined(SSM_FORCE_AVX512)
# define SSM_ARCH SSM_ARCH_AVX512
#elif defined(SSM_FORCE_AVX2)
# define SSM_ARCH SSM_ARCH_AVX2
#elif defined(SSM_FORCE_AVX)
# define SSM_ARCH SSM_ARCH_AVX
#elif defined(SSM_FORCE_SSE4_2)
# define SSM_ARCH SSM_ARCH_SSE4_2
#elif defined(SSM_FORCE_SSE4)
# define SSM_ARCH SSM_ARCH_SSE4
#elif defined(SSM_FORCE_SSSE3)
# define SSM_ARCH SSM_ARCH_SSSE3
#elif defined(SSM_FORCE_SSE3)
# define SSM_ARCH SSM_ARCH_SSE3
#elif defined(SSM_FORCE_SSE2)
# define SSM_ARCH SSM_ARCH_SSE2

#else
# if defined(SSM_COMPILER_CLANG) || defined(SSM_COMPILER_GCC) || (defined(SSM_COMPILER_INTEL) && defined(SSM_PLATFORM_LINUX))
#	if defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512CD__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#	  define SSM_ARCH SSM_ARCH_AVX512
#	elif defined(__AVX2__)
#	  define SSM_ARCH SSM_ARCH_AVX2
#	elif defined(__AVX__)
#	  define SSM_ARCH SSM_ARCH_AVX
#	elif defined(__SSE4_2__)
#	  define SSM_ARCH SSM_ARCH_SSE4_2
#	elif defined(__SSE4_1__)
#	  define SSM_ARCH SSM_ARCH_SSE4
#	elif defined(__SSSE3__)
#	  define SSM_ARCH SSM_ARCH_SSSE3
#	elif defined(__SSE3__)
#	  define SSM_ARCH SSM_ARCH_SSE3
#	elif defined(__SSE2__)
#	  define SSM_ARCH SSM_ARCH_SSE2
# else
#	  define SSM_ARCH SSM_ARCH_PURE
#	endif

# elif defined(SSM_COMPILER_VC) || (defined(SSM_COMPILER_INTEL) && defined(SSM_PLATFORM_WINDOWS))
#	if defined(__AVX2__)
#	  define SSM_ARCH SSM_ARCH_AVX2
#	elif defined(__AVX__)
#	  define SSM_ARCH SSM_ARCH_AVX
#	elif defined(_M_X64)
#	  define SSM_ARCH SSM_ARCH_SSE2
#	elif defined(_M_IX86_FP)
#	  if _M_IX86_FP >= 2
#	  	define SSM_ARCH SSM_ARCH_SSE2
#	  else
#	  	define SSM_ARCH SSM_ARCH_PURE
#	  endif
#	else
#	  define SSM_ARCH SSM_ARCH_PURE
#	endif
# else
#	define SSM_ARCH SSM_ARCH_PURE
#endif
#endif

#if defined(__MINGW64__) && (SSM_ARCH != SSM_ARCH_PURE)
#	include <intrin.h>
#endif

#if SSM_ARCH & SSM_ARCH_AVX2_BIT
#	include <immintrin.h>
#elif SSM_ARCH & SSM_ARCH_AVX_BIT
#	include <immintrin.h>
#elif SSM_ARCH & SSM_ARCH_SSE4_2_BIT
#	include <nmmintrin.h>
#elif SSM_ARCH & SSM_ARCH_SSE4_BIT
#	include <smmintrin.h>
#elif SSM_ARCH & SSM_ARCH_SSSE3_BIT
#	include <tmmintrin.h>
#elif SSM_ARCH & SSM_ARCH_SSE3_BIT
#	include <pmmintrin.h>
#elif SSM_ARCH & SSM_ARCH_SSE2_BIT
#	include <emmintrin.h>
#endif

//---------------------------------------
// Debug macros
//---------------------------------------
#ifdef SSM_FORCE_ENABLE_ASSERTIONS
#undef NDEBUG
#endif
#include <cassert>
