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

#include "../../src/include/hip/detail/api.hpp"
#include "../../src/include/hip/detail/grid_launch.hpp"
#include "../../src/include/hip/detail/helpers.hpp"
#include "../../src/include/hip/detail/intrinsics.hpp"

#include "hip_enums.h"
#include "hip_types.h"

#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>

// BEGIN INTRINSICS
inline
std::uint64_t __ballot(std::int32_t predicate) noexcept
{
    return hip::detail::ballot(predicate);
}

inline
decltype(auto) __clock() noexcept
{
    return hip::detail::clock();
}

inline
decltype(auto) __clock64() noexcept
{
    return hip::detail::clock64();
}

inline
std::uint32_t __clz(std::int32_t x) noexcept
{
    return hip::detail::count_leading_zeroes(x);
}

inline
std::uint32_t __clz(std::uint32_t x) noexcept
{
    return hip::detail::count_leading_zeroes(x);
}

inline
std::uint32_t __clzll(std::int64_t x) noexcept
{
    return hip::detail::count_leading_zeroes(x);
}

inline
std::uint32_t __clzll(std::uint64_t x) noexcept
{
    return hip::detail::count_leading_zeroes(x);
}

inline
std::uint32_t __ffs(std::int32_t x) noexcept
{
    return hip::detail::bit_scan_forward(x);
}

inline
std::uint32_t __ffs(std::uint32_t x) noexcept
{
    return hip::detail::bit_scan_forward(x);
}

inline
std::uint32_t __ffsll(std::int64_t x) noexcept
{
    return hip::detail::bit_scan_forward(x);
}

inline
std::uint32_t __ffsll(std::uint64_t x) noexcept
{
    return hip::detail::bit_scan_forward(x);
}

inline
double __longlong_as_double(std::int64_t x) noexcept
{
    return hip::detail::bit_cast<double>(x);
}

inline
std::uint32_t __popc(std::uint32_t x) noexcept
{
    return hip::detail::pop_count(x);
}

inline
std::uint32_t __popcll(std::uint64_t x) noexcept
{
    return hip::detail::pop_count(x);
}

template<
    typename T,
    std::enable_if_t<
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
inline
T __shfl(T var, std::int32_t src_lane, std::int32_t width = warpSize) noexcept
{
    return hip::detail::shuffle(var, src_lane, width);
}

template<
    typename T,
    std::enable_if_t<
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
inline
T __shfl_down(
    T var, std::uint32_t delta, std::int32_t width = warpSize) noexcept
{
    return hip::detail::shuffle_down(var, delta, width);
}

template<
    typename T,
    std::enable_if_t<
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
inline
T __shfl_up(
    T var, std::uint32_t delta, std::int32_t width = warpSize) noexcept
{
    return hip::detail::shuffle_up(var, delta, width);
}

template<
    typename T,
    std::enable_if_t<
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) >= 4 && sizeof(T) <= 8)>* = nullptr>
inline
T __shfl_xor(
    T var, std::int32_t src_lane, std::int32_t width = warpSize) noexcept
{
    return hip::detail::shuffle_xor(var, src_lane, width);
}

inline
void __syncthreads() noexcept
{
    return hip::detail::Tile::this_tile().barrier();
}

inline
void __threadfence() noexcept
{
    return hip::detail::thread_fence();
}

inline
void __threadfence_block() noexcept
{
    return hip::detail::thread_fence_block();
}

inline
void __threadfence_system() noexcept
{
    return hip::detail::thread_fence_system();
}
// END INTRINSICS

// using hip::detail::clock; // This should get hoovered in via ctime.

inline
decltype(auto) clock64() noexcept
{
    return hip::detail::clock64();
}

inline
hipError_t hipCtxCreate(hipCtx_t* ctx, std::uint32_t flags, hipDevice_t device)
{   // TODO: assess if anything needs to be done for this
    return hip::detail::create_context(ctx, flags, device);
}

inline
hipError_t hipCtxDestroy(hipCtx_t ctx)
{
    return hip::detail::destroy_context(ctx);
}

inline
hipError_t hipDeviceCanAccessPeer(
    std::int32_t* can_access_peer, std::int32_t device_id, std::int32_t peer_id)
{
    return hip::detail::can_access_peer(can_access_peer, device_id, peer_id);
}

inline
hipError_t hipDeviceDisablePeerAccess(std::int32_t peer_device_id)
{
    return hip::detail::disable_peer_access(peer_device_id);
}

inline
hipError_t hipDeviceEnablePeerAccess(
    std::int32_t peer_device_id, std::uint32_t flags)
{
    return hip::detail::enable_peer_access(peer_device_id, flags);
}

inline
hipError_t hipDeviceGet(hipDevice_t* device, std::int32_t device_id)
{
    return hip::detail::get_device(device, device_id);
}

inline
hipError_t hipDeviceReset()
{
    return hip::detail::reset_device();
}

inline
hipError_t hipDeviceSynchronize()
{
    return hip::detail::synchronize_device();
}

inline
hipError_t hipDriverGetVersion(std::int32_t* driver_version)
{
    return hip::detail::get_driver_version(driver_version);
}

inline
hipError_t hipEventCreate(hipEvent_t* event)
{
    return hip::detail::create_event(event);
}

inline
hipError_t hipEventCreateWithFlags(hipEvent_t* event, hipEventFlags flags)
{
    return hip::detail::create_event(event, flags);
}

inline
hipError_t hipEventDestroy(hipEvent_t event)
{
    return hip::detail::destroy_event(event);
}

inline
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)
{
    return hip::detail::delta_time(ms, start, stop);
}

inline
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr)
{
    return hip::detail::insert_event(event, stream);
}

inline
hipError_t hipEventSynchronize(hipEvent_t event)
{
   return hip::detail::wait(event);
}

inline
hipError_t hipFree(void* ptr)
{
    return hip::detail::deallocate(ptr);
}

template<typename F>
inline
hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, F* fn) noexcept
{
    return hip::detail::function_attributes(attr, fn);
}

inline
hipError_t hipGetDevice(std::int32_t* device_id)
{
    return hip::detail::get_device_id(device_id);
}

inline
hipError_t hipDeviceGetLimit(std::size_t* pval, hipLimit_t limit) noexcept
{
    return hip::detail::device_limit(pval, limit);
}

inline
hipError_t hipGetDeviceProperties(
    hipDeviceProp_t* props, std::int32_t device) noexcept
{
    return hip::detail::device_properties(props, device);
}

inline
hipError_t hipGetDeviceCount(std::int32_t* count)
{
    return hip::detail::device_count(count);
}

inline
constexpr
const char* hipGetErrorName(hipError_t error) noexcept
{
    return hip::detail::to_string(error);
}

inline
constexpr
const char* hipGetErrorString(hipError_t error) noexcept
{
    return hip::detail::to_string(error);
}

inline
hipError_t hipGetSymbolAddress(void** out_ptr, const void* symbol) noexcept
{
    return hip::detail::symbol_address(out_ptr, symbol);
}

template<typename T>
inline
hipError_t hipGetSymbolSize(std::size_t* size, const T* symbol) noexcept
{
    return hip::detail::symbol_size(size, symbol);
}

inline
hipError_t hipGetLastError() noexcept
{
    return hip::detail::last_error();
}

inline
hipError_t hipHostGetDevicePointer(
    void** device_ptr, void* host_ptr, unsigned int flags) noexcept
{
    return hip::detail::device_address(device_ptr, host_ptr, flags);
}

inline
hipError_t hipHostFree(void* ptr)
{
    return hip::detail::deallocate_host(ptr);
}

inline
hipError_t hipHostRegister(
    void* ptr, std::size_t size, std::uint32_t flags) noexcept
{
    return hip::detail::lock_memory(ptr, size, flags);
}

inline
hipError_t hipHostUnregister(void* ptr) noexcept
{
    return hip::detail::unlock_memory(ptr);
}

inline
hipError_t hipHostMalloc(
    void** ptr,
    std::size_t size,
    hipHostMallocKind flags = hipHostMallocDefault)
{
    return hip::detail::allocate_host(ptr, size, flags);
}

inline
hipError_t hipInit(std::uint32_t flags) noexcept
{
    return hip::detail::init_rt(flags);
}

#define hipLaunchKernelGGL(\
    kernel_name, num_blocks, dim_blocks, group_mem_bytes, stream, ...)\
    if (true) {\
        ::hip::detail::launch(\
            num_blocks,\
            dim_blocks,\
            group_mem_bytes,\
            stream,\
            [=](auto&&... xs) noexcept {\
                kernel_name(std::forward<decltype(xs)>(xs)...);\
        }, std::make_tuple(__VA_ARGS__));\
    }\
    else ((void)0)

inline
hipError_t hipMalloc(void** ptr, std::size_t size)
{
    return hip::detail::allocate(ptr, size);
}

inline
hipError_t hipMallocManaged(
    void** ptr,
    std::size_t size,
    hipMallocManagedFlags /*flags*/ = hipMemAttachHost)
{
    return hip::detail::allocate(ptr, size);
}

inline
hipError_t hipMemcpy(
    void* dst,
    const void* src,
    std::size_t size,
    hipMemcpyKind kind = hipMemcpyDefault)
{
    return hip::detail::copy(dst, src, size, kind);
}

inline
hipError_t hipMemcpy2DAsync(
    void* dst,
    std::size_t d_pitch,
    const void* src,
    std::size_t s_pitch,
    std::size_t width,
    std::size_t height,
    hipMemcpyKind kind = hipMemcpyDefault,
    hipStream_t stream = nullptr)
{
    return hip::detail::copy_2d_async(
        dst, d_pitch, src, s_pitch, width, height, kind, stream);
}

inline
hipError_t hipMemcpyAsync(
    void* dst,
    const void* src,
    std::size_t size,
    hipMemcpyKind kind = hipMemcpyDefault,
    hipStream_t stream = nullptr)
{
    return hip::detail::copy_async(dst, src, size, kind, stream);
}

inline
hipError_t hipMemcpyDtoH(void* dst, const void* src, std::size_t size)
{
    return hip::detail::copy(dst, src, size, hipMemcpyDeviceToHost);
}

inline
hipError_t hipMemcpyFromSymbol(
    void* dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyDeviceToHost)
{
    return hip::detail::copy_to_symbol(dst, src, byte_cnt, offset, kind);
}

inline
hipError_t hipMemcpyFromSymbolAsync(
    void* dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyDeviceToHost,
    hipStream_t stream = nullptr)
{
    return hip::detail::copy_to_symbol_async(
        dst, src, byte_cnt, offset, kind, stream);
}

inline
hipError_t hipMemcpyHtoD(void* dst, const void* src, std::size_t size)
{
    return hip::detail::copy(dst, src, size, hipMemcpyHostToDevice);
}

inline
hipError_t hipMemcpyToSymbol(
    void* dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyHostToDevice)
{
    return hip::detail::copy_to_symbol(dst, src, byte_cnt, offset, kind);
}

template<typename T>
[[deprecated("This is a HIP specific extension and thus non-portable.")]]
inline
hipError_t hipMemcpyToSymbol(
    T& dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyHostToDevice)
{
    return hipMemcpyToSymbol(
        reinterpret_cast<void*>(&dst), src, byte_cnt, offset, kind);
}

inline
hipError_t hipMemcpyToSymbolAsync(
    void* dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyHostToDevice,
    hipStream_t stream = nullptr)
{
    return hip::detail::copy_to_symbol_async(
        dst, src, byte_cnt, offset, kind, stream);
}

template<typename T>
[[deprecated("This is a HIP specific extension and thus non-portable.")]]
inline
hipError_t hipMemcpyToSymbolAsync(
    T& dst,
    const void* src,
    std::size_t byte_cnt,
    std::size_t offset = 0,
    hipMemcpyKind kind = hipMemcpyHostToDevice,
    hipStream_t stream = nullptr)
{
    return hipMemcpyToSymbolAsync(
        reinterpret_cast<void*>(&dst), src, byte_cnt, offset, kind, stream);
}

inline
hipError_t hipMemGetInfo(std::size_t* available, std::size_t* total) noexcept
{
    return hip::detail::memory_info(available, total);
}

inline
hipError_t hipMemset(void* dst, int value, std::size_t size_bytes)
{
    return hip::detail::fill_bytes(dst, value, size_bytes);
}

inline
hipError_t hipModuleGetFunction(
    hipFunction_t* f, hipModule_t m, const char* function_name)
{
    return hip::detail::get_function(f, m, function_name);
}

inline
hipError_t hipModuleGetGlobal(
    hipDeviceptr_t* pp,
    std::size_t* size_bytes,
    hipModule_t m,
    const char* name)
{
    return hip::detail::get_global(pp, size_bytes, m, name);
}

inline
hipError_t hipModuleLaunchKernel(
    hipFunction_t gfn,
    std::uint32_t grid_dim_x,
    std::uint32_t grid_dim_y,
    std::uint32_t grid_dim_z,
    std::uint32_t block_dim_x,
    std::uint32_t block_dim_y,
    std::uint32_t block_dim_z,
    std::uint32_t shared_mem_bytes,
    hipStream_t stream,
    void** kernel_params,
    void** extra)
{
    return hip::detail::launch_kernel_from_so(
        gfn,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_bytes,
        stream,
        kernel_params,
        extra);
}

inline
hipError_t hipModuleLoad(hipModule_t* m, const char* file_name)
{
    return hip::detail::load_module(m, file_name);
}

template<typename F>
inline
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    std::int32_t* num_blocks,
    F f,
    std::int32_t block_size,
    std::size_t dynamic_shared_memory_per_block)
{
    return hip::detail::max_blocks_per_sm(
        num_blocks, f, block_size, dynamic_shared_memory_per_block);
}

template<typename F>
inline
hipError_t hipOccupancyMaxPotentialBlockSize(
    std::uint32_t* grid_size,
    std::uint32_t* block_size,
    F f,
    std::size_t dynamic_shared_memory_per_block,
    std::uint32_t block_size_limit)
{
    return hip::detail::max_potential_block_size(
        grid_size,
        block_size,
        f,
        dynamic_shared_memory_per_block,
        block_size_limit);
}

inline
hipError_t hipPeekAtLastError() noexcept
{
    return hip::detail::peek_error();
}

inline
hipError_t hipSetDevice(std::int32_t device_id) noexcept
{
    return hip::detail::set_device(device_id);
}

inline
hipError_t hipSetDeviceFlags(std::int32_t flags) noexcept
{
    return hip::detail::set_device_flags(flags);
}

inline
hipError_t hipStreamCreate(hipStream_t* stream)
{
    return hip::detail::create_stream(stream);
}

inline
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, hipStreamFlags flags)
{
    return hip::detail::create_stream_with_flags(stream, flags);
}

inline
hipError_t hipStreamDestroy(hipStream_t stream)
{
    return hip::detail::destroy_stream(stream);
}

inline
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    return hip::detail::synchronize_stream(stream);
}

inline
hipError_t hipStreamWaitEvent(
    hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    return hip::detail::stream_wait_for(stream, event, flags);
}