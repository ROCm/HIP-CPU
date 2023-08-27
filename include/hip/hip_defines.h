/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#include <version>
#if !defined(__cpp_lib_parallel_algorithm)
    #error The HIP-CPU RT requires a C++17 compliant standard library which exposes parallel algorithms support; see https://en.cppreference.com/w/cpp/algorithm#Execution_policies.
#endif

#define __HIP_CPU_RT__

#include <cstddef>
#include <utility>
#include <vector>

#define __HIP__ // TODO: potentially dangerous.
#define __HIPCC__ // TODO: potentially dangerous.

#if defined(__clang__) || defined(__GNUC__)
    #define __constant__ /*constexpr*/
#else
    #define __constant__ __declspec(dllexport)
#endif

#if defined(__clang__) || defined(_MSC_VER)
    #define __device__ /*_Pragma("omp declare simd")*/
#else
    #define __device__ __attribute__((simd))
#endif
#if defined(NDEBUG)
    #if defined(__clang__) || defined(__GNUC__)
        #define __forceinline__ __attribute__((always_inline)) inline
    #else
        #if defined(__has_cpp_attribute)
            #if __has_cpp_attribute(msvc::forceinline)
                #define __forceinline__ [[msvc::forceinline]] inline
            #else
                #define __forceinline__ __forceinline
            #endif
        #else
            #define __forceinline__ __forceinline
        #endif
    #endif
#else
    #define __forceinline__
#endif

#if defined(_WIN32)
    #if !defined(HIP_LIBRARY_BUILD)
        #define __HIP_API__ __declspec(dllexport)
    #else
        #define __HIP_API__ __declspec(dllimport)
    #endif
#else
    #define __HIP_API__ __attribute__((visibility("default")))
#endif

#define __host__

#if defined(__clang__)
    #define __global__ __attribute__((flatten))//, vectorcall))
#elif defined(__GNUC__)
    #if __GNUC__ != 10
        #define __global__ __attribute__((flatten, simd))
    #else
        #define __global__ __attribute__((flatten))
    #endif
#elif defined(_MSC_VER)
    #define __global__ __forceinline__ __declspec(dllexport)
#endif

#define HIP_KERNEL_NAME(...) __VA_ARGS__

#define __launch_bounds__(...)

#define HIP_DYNAMIC_SHARED(type, variable)\
    thread_local static std::vector<std::byte> __hip_##variable##_storage__;\
    __hip_##variable##_storage__.resize(\
        scratchpad_size(domain(::hip::detail::Tile::this_tile())));\
    auto variable{\
        reinterpret_cast<type*>(std::data(__hip_##variable##_storage__))};
#define __shared__ thread_local

#define HIP_SYMBOL(X) &X

// TODO: temporary
#if defined(__has_cpp_attribute)
    #if __has_cpp_attribute(gnu::flatten)
        #define __HIP_FLATTENED_FUNCTION__ [[gnu::flatten]]
    #else
        #define __HIP_FLATTENED_FUNCTION__
    #endif
#endif

#if defined(_MSC_VER)
    #define __HIP_TILE_FUNCTION__ __forceinline__
#elif defined(__clang__)
    #define __HIP_TILE_FUNCTION__\
        _Pragma("omp declare simd")\
        __attribute__((always_inline, flatten, vectorcall))
#elif defined(__GNUC__)
    #define __HIP_TILE_FUNCTION__ __attribute__((always_inline, flatten, simd))
#endif

#if defined(_MSC_VER)  // TODO: revisit when _Pragma is supported
    #define __HIP_VECTORISED_LOOP__ __pragma(omp simd)
#else
    #define __HIP_VECTORISED_LOOP__ _Pragma("omp simd")
#endif
