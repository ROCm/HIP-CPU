/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#pragma once

#if !defined(__HIP_CPU_RT__)
    #error Private HIP-CPU RT implementation headers must not be included directly.
#endif

#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "function.hpp"
#include "grid_launch.hpp"
#include "module.hpp"
#include "runtime.hpp"
#include "system.hpp"
#include "../../../../include/hip/hip_constants.h"
#include "../../../../include/hip/hip_enums.h"
#include "../../../../include/hip/hip_types.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <utility>
#include <cstring>

namespace hip
{
    namespace detail
    {
        inline
        hipError_t allocate(void** p, std::size_t byte_cnt)
        {
            if (!p) return hipErrorInvalidValue;

            *p = std::malloc(byte_cnt);

            if (!p) return hipErrorOutOfMemory;

            return hipSuccess;
        }

        inline
        hipError_t allocate_host(
            void** p, std::size_t byte_cnt, hipHostMallocKind /*flags*/)
        {
            return allocate(p, byte_cnt);
        }

        inline
        hipError_t allocate_pitch(
            void**  p,
            std::size_t* pitch,
            std::size_t width,
            std::size_t height)
        {
            if (!p) return hipErrorInvalidValue;

            *pitch = width;
            *p = std::malloc(width * height);

            if (!p) return hipErrorOutOfMemory;

            return hipSuccess;
        }

        inline
        hipError_t copy_async(
            void* dst,
            const void* src,
            std::size_t size,
            hipMemcpyKind /*kind*/,
            Stream* s)
        {
            if (size == 0) return hipSuccess;
            if (!dst || !src) return hipErrorInvalidValue;

            if (!s) s = Runtime::null_stream();

            s->apply([=](auto&& ts) {
                ts.emplace_back([=](auto&&) { std::memcpy(dst, src, size); });
            });

            return hipSuccess;
        }

        hipError_t synchronize_device(); // Forward declaration.
        hipError_t synchronize_stream(Stream*); // Forward declaration.

        inline
        hipError_t copy(
            void* dst,
            const void* src,
            std::size_t byte_cnt,
            hipMemcpyKind kind)
        {
            if (byte_cnt == 0) return hipSuccess;
            if (!dst || !src) return hipErrorInvalidValue;

            synchronize_device();

            const auto s{copy_async(dst, src, byte_cnt, kind, nullptr)};

            if (s != hipSuccess) return s;

            return synchronize_stream(nullptr);
        }

        inline
        hipError_t copy_2d_async(
            void* dst,
            std::size_t d_pitch,
            const void* src,
            std::size_t s_pitch,
            std::size_t width,
            std::size_t height,
            hipMemcpyKind /*kind*/,
            Stream* s)
        {
            if (height * width == 0) return hipSuccess;
            if (!dst || !src) return hipErrorInvalidValue;

            if (!s) s = Runtime::null_stream();

            s->apply([=](auto&& ts) { // TODO: optimise.
                ts.emplace_back(
                    [=,
                    dst = static_cast<std::byte*>(dst),
                    src = static_cast<const std::byte*>(src)](auto&&) mutable {
                    for (auto i = 0u; i != height; ++i) {
                        std::memcpy(dst, src, width);

                        dst += d_pitch;
                        src += s_pitch;
                    }
                });
            });

            return hipSuccess;
        }

        inline
        hipError_t copy_2d(
            void* dst,
            std::size_t d_pitch,
            const void* src,
            std::size_t s_pitch,
            std::size_t width,
            std::size_t height,
            hipMemcpyKind kind)
        {
            if (height * width == 0) return hipSuccess;
            if (!dst || !src) return hipErrorInvalidValue;

            synchronize_device();

            const auto s{copy_2d_async(
                dst, d_pitch, src, s_pitch, width, height, kind, nullptr)};

            if (s != hipSuccess) return s;

            return synchronize_stream(nullptr);
        }

        inline
        hipError_t copy_to_symbol(
            void* dst,
            const void* src,
            std::size_t byte_cnt,
            std::size_t dx,
            hipMemcpyKind kind)
        {
            if (byte_cnt == 0) return hipSuccess;
            if (!dst) return hipErrorInvalidSymbol;
            if (!src) return hipErrorInvalidValue;

            return copy(static_cast<std::byte*>(dst) + dx, src, byte_cnt, kind);
        }

        inline
        hipError_t copy_to_symbol_async(
            void* dst,
            const void* src,
            std::size_t byte_cnt,
            std::size_t dx,
            hipMemcpyKind kind,
            Stream* stream)
        {
            if (byte_cnt == 0) return hipSuccess;
            if (!dst) return hipErrorInvalidSymbol;
            if (!src) return hipErrorInvalidValue;

            return copy_async(
                static_cast<std::byte*>(dst) + dx,
                src,
                byte_cnt,
                kind,
                stream);
        }

        inline
        hipError_t create_context(
            Context** ctx, std::uint32_t /*flags*/, Device /*device*/)
        {   // TODO: assess if anything needs to be done for this
            if (!ctx) return hipErrorInvalidValue;

            *ctx = nullptr;

            return hipSuccess;
        }

        inline
        hipError_t create_event(Event** event)
        {
            if (!event) return hipErrorInvalidValue;

            *event = new Event;

            if (!*event) return hipErrorOutOfMemory;

            return hipSuccess;
        }

        inline
        hipError_t create_event(Event** event, hipEventFlags flags)
        {
            if (!event) return hipErrorInvalidValue;

            *event = new hip::detail::Event{flags};

            if (!*event) return hipErrorOutOfMemory;

            return hipSuccess;
        }

        inline
        hipError_t create_stream(Stream** s)
        {
            if (!s) return hipErrorInvalidValue;

            *s = Runtime::make_stream_async().get();

            return *s ? hipSuccess : hipErrorRuntimeOther;
        }

        inline
        hipError_t create_stream_with_flags(Stream** s, hipStreamFlags)
        {
            return create_stream(s);
        }

        inline
        hipError_t can_access_peer(
            std::int32_t* can_access,
            std::int32_t /*d_id*/,
            std::int32_t /*p_id*/)
        {
            if (!can_access) return hipErrorInvalidValue;

            *can_access = 0;

            return hipSuccess;
        }

        inline
        hipError_t deallocate(void* p)
        {
            synchronize_device();

            if (p) std::free(p);

            return hipSuccess;
        }

        inline
        hipError_t deallocate_host(void* p)
        {
            return hip::detail::deallocate(p);
        }

        inline
        hipError_t delta_time(float* ms, Event* t0, Event* t1)
        {
            if (!ms) return hipErrorInvalidValue;
            if (!t0) return hipErrorInvalidHandle;
            if (!t1) return hipErrorInvalidHandle;

            if (!is_done(*t0).valid()) return hipErrorInvalidHandle;
            if (!is_done(*t1).valid()) return hipErrorInvalidHandle;

            is_done(*t1).wait();
            is_done(*t0).wait();

            using MS =
                std::chrono::duration<float, std::chrono::milliseconds::period>;
            *ms = std::chrono::duration_cast<MS>(
                timestamp(*t1) - timestamp(*t0)).count();

            return hipSuccess;
        }

        inline
        hipError_t destroy_context(Context* ctx)
        {   // TODO: assess if anything needs to be done for this
            if (ctx) return hipErrorInvalidValue; // Impossible to create one.

            return hipSuccess;
        }

        inline
        hipError_t destroy_event(Event* event)
        {
            if (!event) return hipErrorInvalidHandle;

            delete event;

            return hipSuccess;
        }

        inline
        hipError_t destroy_stream(Stream* s)
        {
            if (!s) return hipErrorInvalidHandle;

            Runtime::destroy_stream_async(s).wait();

            return hipSuccess;
        }

        inline
        hipError_t device_address(
            void** pd, void* ph, unsigned int/* flags */) noexcept
        {
            if (!pd) return hipErrorInvalidValue;
            if (!ph) return hipErrorInvalidValue;

            *pd = ph;

            return hipSuccess;
        }

        inline
        hipError_t device_count(std::int32_t* n)
        {
            if (!n) return hipErrorInvalidValue;

            *n = 1; // TODO: enumerate packages?

            // if (*count == 0) return hipErrorNoDevice;

            return hipSuccess;
        }

        inline
        hipError_t device_limit(std::size_t* px, hipLimit_t l) noexcept
        {
            if (!px) return hipErrorInvalidValue;
            if (l != hipLimitMallocHeapSize) return hipErrorUnsupportedLimit;

            *px = System::memory().total;

            return hipSuccess;
        }

        inline
        hipError_t device_properties(
            hipDeviceProp_t* p, std::int32_t d_id) noexcept
        {
            if (!p) return hipErrorInvalidValue;
            if (d_id != 0) return hipErrorInvalidDevice; // Only 1 device for now.

            const auto n{System::cpu_name()};
            const auto it{std::copy_n(
                std::cbegin(n),
                std::min(std::size(n), std::size(p->name) - 1),
                p->name)};
            *it = '\0';

            p->totalGlobalMem = System::memory().total;
            p->sharedMemPerBlock = System::cpu_cache().l1.size; // L1 ~= LDS

            switch (System::architecture()) { // regsPerBlock ~= architectural Regs
            case System::Architecture::x86: p->regsPerBlock = 8; break;
            case System::Architecture::x64: p->regsPerBlock = 16; break;
            case System::Architecture::arm: p->regsPerBlock = 16; break;
            case System::Architecture::arm64: p->regsPerBlock = 16; break;
            default: p->regsPerBlock = 8;
            }

            p->warpSize = System::simd_width();
            p->maxThreadsPerBlock = 1024; // Arbitrary.
            std::fill_n(p->maxThreadsDim, std::size(p->maxThreadsDim), 1024);
            std::fill_n(p->maxGridSize, std::size(p->maxGridSize), INT32_MAX);
            p->clockRate = System::cpu_frequency() * 1000;
            p->memoryClockRate = 0; // TODO
            p->memoryBusWidth = 0; // TODO
            p->totalConstMem = p->totalGlobalMem;
            p->major = 7;
            p->minor = 0;
            p->multiProcessorCount = System::core_count();
            p->l2CacheSize = System::cpu_cache().l2.size;
            p->maxThreadsPerMultiProcessor =
                std::thread::hardware_concurrency() * p->maxThreadsPerBlock;
            p->computeMode = 1; // TODO
            p->clockInstructionRate = static_cast<std::int32_t>(
                std::chrono::high_resolution_clock::period::den /
                    (std::chrono::high_resolution_clock::period::num * 1000.0));

            p->arch.hasGlobalInt32Atomics = 1;
            p->arch.hasGlobalFloatAtomicExch = 1;
            p->arch.hasSharedInt32Atomics = 1;
            p->arch.hasSharedFloatAtomicExch = 1;
            p->arch.hasFloatAtomicAdd = 0;
            p->arch.hasGlobalInt64Atomics = 1;
            p->arch.hasSharedInt64Atomics = 1;
            p->arch.hasDoubles = 1;
            p->arch.hasWarpVote = 1;
            p->arch.hasWarpBallot = 1;
            p->arch.hasWarpShuffle = 1;
            p->arch.hasFunnelShift = 1;
            p->arch.hasThreadFenceSystem = 1;
            p->arch.hasSyncThreadsExt = 0;
            p->arch.hasSurfaceFuncs = 0;
            p->arch.has3dGrid = 1;
            p->arch.hasDynamicParallelism = 1;
            p->concurrentKernels = 1;
            p->pciDomainID = 0;
            p->pciBusID = 0;
            p->pciDeviceID = 0;
            p->maxSharedMemoryPerMultiProcessor = System::cpu_cache().l2.size;
            p->isMultiGpuBoard = 0;
            p->canMapHostMemory = 1;
            p->gcnArch = 69;
            p->integrated = 1;
            p->cooperativeLaunch = 1;
            p->cooperativeMultiDeviceLaunch = 1;
            p->maxTexture1D = 0;
            std::fill_n(p->maxTexture2D, std::size(p->maxTexture2D), 0);
            std::fill_n(p->maxTexture3D, std::size(p->maxTexture3D), 0);
            p->hdpMemFlushCntl = nullptr;
            p->hdpRegFlushCntl = nullptr;
            p->memPitch = 0;
            p->textureAlignment = 0;
            p->texturePitchAlignment = 0;
            p->kernelExecTimeoutEnabled = 0;
            p->ECCEnabled = 0; // TODO
            p->tccDriver = 0;
            p->cooperativeMultiDeviceUnmatchedFunc = 1;
            p->cooperativeMultiDeviceUnmatchedGridDim = 1;
            p->cooperativeMultiDeviceUnmatchedBlockDim = 1;
            p->cooperativeMultiDeviceUnmatchedSharedMem = 1;
            p->deviceOverlap = 1;

            return hipSuccess;
        }

        inline
        hipError_t disable_peer_access(std::int32_t d_id) noexcept
        {
            return d_id ? hipSuccess : hipErrorInvalidDevice;
        }

        inline
        hipError_t enable_peer_access(
            std::int32_t p_id, std::uint32_t flags) noexcept
        {
            if (flags) return hipErrorInvalidValue;

            return (p_id != 0) ? hipSuccess : hipErrorInvalidDevice;
        }

        template<typename T>
        inline
        hipError_t fill_n_async(T* p, std::size_t n, T x, Stream* s)
        {
            if (n == 0) return hipSuccess;
            if (!p) return hipErrorInvalidValue;

            if (!s) s = Runtime::null_stream();

            s->apply([=](auto&& ts) {
                ts.emplace_back([=](auto&&) { std::fill_n(p, n, x); });
            });

            return hipSuccess;
        }

        template<typename T>
        inline
        hipError_t fill_n(T* p, std::size_t n, T x)
        {
            if (n == 0) return hipSuccess;
            if (!p) return hipErrorInvalidValue;

            synchronize_device();

            const auto s{fill_n_async(p, n, x, Runtime::null_stream())};

            if (s != hipSuccess) return s;

            return synchronize_stream(Runtime::null_stream());

            return hipSuccess;
        }

        template<typename F>
        inline
        hipError_t function_attributes(hipFuncAttributes* pa, F* fn) noexcept
        {
            if (!pa) return hipErrorInitializationError;
            if (!fn) return hipErrorInvalidDeviceFunction;

            static const auto props{[]() {
                hipDeviceProp_t r{};
                device_properties(&r, 0); // Only 1 device.

                return r;
            }()};

            pa->constSizeBytes = 0;
            pa->localSizeBytes = 0;
            pa->maxThreadsPerBlock = props.maxThreadsPerBlock;
            pa->numRegs = props.regsPerBlock;
            pa->sharedSizeBytes = 0;

            return hipSuccess;
        }

        inline
        hipError_t get_device(Device* device, std::int32_t /*device_id*/)
        {
            *device = 0; // Only 1 device.

            return hipSuccess;
        }

        inline
        hipError_t get_device_id(std::int32_t* d_id) noexcept
        {
            if (!d_id) return hipErrorInvalidValue;

            *d_id = 0; // Only 1 device.

            return hipSuccess;
        }

        inline
        hipError_t get_driver_version(std::int32_t* driver_version) noexcept
        {
            if (!driver_version) return hipErrorInvalidValue;

            *driver_version = 0;

            return hipSuccess;
        }

        inline
        hipError_t get_function(Function** f, Module* m, const char* fn)
        {   // TODO: re-use hipModuleGetGlobal
            if (!f) return hipErrorInvalidValue;
            if (!m) return hipErrorInvalidHandle;
            if (!valid(*m)) return hipErrorInvalidHandle;
            if (!fn) return hipErrorInvalidValue;

            *f = new Function{*m, fn};

            if (!*f) return hipErrorOutOfMemory;
            if (!valid(**f)) return hipErrorNotFound;

            return hipSuccess;
        }

        inline
        hipError_t get_global(
            void** pp, std::size_t* size_bytes, Module* m, const char* name)
        {
            if (!pp) return hipErrorInvalidValue;
            if (!size_bytes) return hipErrorInvalidValue;
            if (!m || !valid(*m)) return hipErrorInvalidImage;
            if (!name) return hipErrorInvalidValue;

            *pp = address_of(*m, name);

            if (!*pp) return hipErrorNotFound;

            *size_bytes = 0; // Unsupported for now, requires ELF / COFF / MACHO introspection.

            return hipSuccess;
        }

        inline
        hipError_t init_rt(std::uint32_t flags) noexcept
        {
            return (flags == 0u) ? hipSuccess : hipErrorInvalidValue;
        }

        inline
        hipError_t insert_event(Event* px, Stream* into)
        {
            if (!px) return hipErrorInvalidHandle;

            Runtime::push_task(px, into);

            return hipSuccess;
        }

        inline
        hipError_t last_error() noexcept
        {
            return Runtime::set_last_error(hipSuccess);
        }

        inline
        hipError_t launch_kernel_from_so(
            Function* gfn,
            std::uint32_t gd_x,
            std::uint32_t gd_y,
            std::uint32_t gd_z,
            std::uint32_t bd_x,
            std::uint32_t bd_y,
            std::uint32_t bd_z,
            std::uint32_t shm_bytes,
            Stream* stream,
            void** kernel_params,
            void** extra)
        {
            if (!gfn) return hipErrorInvalidValue;
            if (kernel_params) throw std::runtime_error{"Unimplemented."};

            if (gd_x > UINT32_MAX) return hipErrorInvalidConfiguration;
            if (gd_y > UINT32_MAX) return hipErrorInvalidConfiguration;
            if (gd_z > UINT32_MAX) return hipErrorInvalidConfiguration;

            if (bd_x * bd_y * bd_z > 1024) { // TODO: remove magic constant.
                return hipErrorInvalidConfiguration;
            }

            if (shm_bytes > 64 * 1024) return hipErrorInvalidConfiguration; // TODO: remove magic constant.

            const auto tile_ctx_addr{reinterpret_cast<void* (*)()>(
                address_of(parent(*gfn),
                "_hip_detail_this_tile_so_local_context"))};

            const auto& cfg{reinterpret_cast<void** (&)[5]>(*extra)};
            const auto arg_ptr{reinterpret_cast<std::byte*>(cfg[1])};
            const auto arg_sz{*reinterpret_cast<std::size_t*>(cfg[3])};

            std::vector<std::byte> args{arg_ptr, arg_ptr + arg_sz};
            launch(
                {gd_x, gd_y, gd_z},
                {bd_x, bd_y, bd_z},
                shm_bytes,
                stream,
                [=, args = std::move(args)]() {
                *static_cast<Tile*>(tile_ctx_addr()) = Tile::this_tile();

                native_handle(*gfn)();
            }, {});

            return arg_sz ? hipErrorInvalidImage : hipSuccess; // Calling functions taking arguments unsupported.
        }

        inline
        hipError_t load_module(Module** m, const char* so_name)
        {
            if (!m) return hipErrorInvalidValue;
            if (!so_name) return hipErrorInvalidValue;
            if (!std::filesystem::exists(so_name)) return hipErrorFileNotFound;

            *m = new Module{so_name};

            if (!*m) return hipErrorOutOfMemory;
            if (!valid(**m)) return hipErrorSharedObjectInitFailed;

            return hipSuccess;
        }

        inline
        hipError_t lock_memory(
            void* p, std::size_t byte_cnt, std::uint32_t /*flags*/) noexcept
        {
            if (!p) return hipErrorInvalidValue;
            if (byte_cnt == 0) return hipErrorInvalidValue;

            return hipSuccess;
        }

        template<typename F>
        inline
        hipError_t max_blocks_per_sm(
            std::int32_t* n,
            F /*f*/,
            std::uint32_t /*block_sz*/,
            std::size_t /*dynamic_shm_per_block*/)
        {
            if (!n) return hipErrorInvalidValue;

            static const auto props{[]() {
                hipDeviceProp_t r{};
                device_properties(&r, 0); // Only 1 device.

                return r;
            }()};

            *n = props.maxThreadsPerMultiProcessor;

            return hipSuccess;
        }

        template<typename F>
        inline
        hipError_t max_potential_block_size(
            std::uint32_t* g_sz,
            std::uint32_t* b_sz,
            F /*f*/, // No metadata to introspect.
            std::size_t /*dynamic_shm_per_block*/,
            std::uint32_t max_b_sz)
        {   // TODO: mostly arbitrary, should probably issue more blocks per thread;
            if (!g_sz) return hipErrorInvalidValue;
            if (!b_sz) return hipErrorInvalidValue;

            static const auto props{[]() {
                hipDeviceProp_t r{};
                device_properties(&r, 0); // Only 1 device.

                return r;
            }()};

            *g_sz = std::thread::hardware_concurrency();
            *b_sz = max_b_sz ?
                std::min<std::uint32_t>(max_b_sz, props.maxThreadsPerBlock) :
                props.maxThreadsPerBlock;

            return hipSuccess;
        }

        inline
        hipError_t memory_info(std::size_t* pa, std::size_t* pt)
        {
            if (!pa) return hipErrorInvalidValue;
            if (!pt) return hipErrorInvalidValue;

            *pa = System::memory().available;
            *pt = System::memory().total;

            return hipSuccess;
        }

        inline
        hipError_t peek_error() noexcept
        {
            return Runtime::last_error();
        }

        inline
        hipError_t reset_device() noexcept
        {
            return hipSuccess;
        }

        inline
        hipError_t runtime_version(std::int32_t* pv) noexcept
        {   // TODO: this is a placeholder spoofing 5.6.
            if (!pv) return hipErrorInvalidValue;

            constexpr auto hip_major{5};
            constexpr auto hip_minor{6};
            constexpr auto hip_patch{0};

            *pv = hip_major * 10000000 + hip_minor * 100000 + hip_patch;

            return hipSuccess;
        }

        inline
        hipError_t set_device(std::int32_t d_id) noexcept
        {
            return (d_id == 0) ? hipSuccess : hipErrorInvalidDevice; // Only 1 device.
        }

        inline
        hipError_t set_device_flags(std::int32_t f) noexcept
        {   // TODO: ignored for now, but useful for picking inline execution.
            if (f < hipDeviceScheduleAuto) return hipErrorInvalidValue;
            if (f > hipDeviceLmemResizeToMax) return hipErrorInvalidValue;
            return hipSuccess;
        }

        extern hipError_t wait(Event*); // Forward declaration.

        inline
        hipError_t stream_wait_for(
            Stream* s, Event* e, unsigned int /*flags*/)
        {
            if (!e) return hipErrorInvalidHandle;

            if (!s) s = Runtime::null_stream();

            s->apply([=](auto&& ts) {
                ts.emplace_back([=](auto&&) { wait(e); });
            });

            return hipSuccess;
        }

        inline
        hipError_t symbol_address(void** p, const void* s) noexcept
        {
            if (!p || !s) return hipErrorInvalidValue;

            *p = const_cast<void*>(s);

            return hipSuccess;
        }

        template<typename T>
        inline
        hipError_t symbol_size(std::size_t* sz, const T* s) noexcept
        {
            if (!sz || !s) return hipErrorInvalidValue;

            *sz = sizeof(T);

            return hipSuccess;
        }

        inline
        hipError_t synchronize_device()
        {
            Runtime::synchronize();

            return hipSuccess;
        }

        extern hipError_t wait(Event*); // Forward declaration.

        inline
        hipError_t synchronize_stream(Stream* s)
        {
            Event tmp{};

            if (insert_event(&tmp, s) != hipSuccess) return hipErrorUnknown;
            if (wait(&tmp) != hipSuccess) return hipErrorUnknown;

            return hipSuccess;
        }

        inline
        constexpr
        const char* to_string(hipError_t e) noexcept
        {
            switch (e) {
            case hipSuccess:
                return "hipSuccess";
            case hipErrorInvalidValue:
                return "hipErrorInvalidValue";
            case hipErrorOutOfMemory:
                return "hipErrorOutOfMemory";
            //case hipErrorMemoryAllocation: return "hipErrorMemoryAllocation";
            case hipErrorNotInitialized:
                return "hipErrorNotInitialized";
            //case hipErrorInitializationError: return "hipErrorInitializationError";
            case hipErrorDeinitialized:
                return "hipErrorDeinitialized";
            case hipErrorProfilerDisabled:
                return "hipErrorProfilerDisabled";
            case hipErrorProfilerNotInitialized:
                return "hipErrorProfilerNotInitialized";
            case hipErrorProfilerAlreadyStarted:
                return "hipErrorProfilerAlreadyStarted";
            case hipErrorProfilerAlreadyStopped:
                return "hipErrorProfilerAlreadyStopped";
            case hipErrorInvalidConfiguration:
                return "hipErrorInvalidConfiguration";
            case hipErrorInvalidSymbol:
                return "hipErrorInvalidSymbol";
            case hipErrorInvalidDevicePointer:
                return "hipErrorInvalidDevicePointer";
            case hipErrorInvalidMemcpyDirection:
                return "hipErrorInvalidMemcpyDirection";
            case hipErrorInsufficientDriver:
                return "hipErrorInsufficientDriver";
            case hipErrorMissingConfiguration:
                return "hipErrorMissingConfiguration";
            case hipErrorPriorLaunchFailure:
                return "hipErrorPriorLaunchFailure";
            case hipErrorInvalidDeviceFunction:
                return "hipErrorInvalidDeviceFunction";
            case hipErrorNoDevice:
                return "hipErrorNoDevice";
            case hipErrorInvalidDevice:
                return "hipErrorInvalidDevice";
            case hipErrorInvalidImage:
                return "hipErrorInvalidImage";
            case hipErrorInvalidContext:
                return "hipErrorInvalidContext";
            case hipErrorContextAlreadyCurrent:
                return "hipErrorContextAlreadyCurrent";
            case hipErrorMapFailed:
                return "hipErrorMapFailed";
            //case hipErrorMapBufferObjectFailed: return "hipErrorMapBufferObjectFailed";
            case hipErrorUnmapFailed:
                return "hipErrorUnmapFailed";
            case hipErrorArrayIsMapped:
                return "hipErrorArrayIsMapped";
            case hipErrorAlreadyMapped:
                return "hipErrorAlreadyMapped";
            case hipErrorNoBinaryForGpu:
                return "hipErrorNoBinaryForGpu";
            case hipErrorAlreadyAcquired:
                return "hipErrorAlreadyAcquired";
            case hipErrorNotMapped:
                return "hipErrorNotMapped";
            case hipErrorNotMappedAsArray:
                return "hipErrorNotMappedAsArray";
            case hipErrorNotMappedAsPointer:
                return "hipErrorNotMappedAsPointer";
            case hipErrorECCNotCorrectable:
                return "hipErrorECCNotCorrectable";
            case hipErrorUnsupportedLimit:
                return "hipErrorUnsupportedLimit";
            case hipErrorContextAlreadyInUse:
                return "hipErrorContextAlreadyInUse";
            case hipErrorPeerAccessUnsupported:
                return "hipErrorPeerAccessUnsupported";
            case hipErrorInvalidKernelFile:
                return "hipErrorInvalidKernelFile";
            case hipErrorInvalidGraphicsContext:
                return "hipErrorInvalidGraphicsContext";
            case hipErrorInvalidSource:
                return "hipErrorInvalidSource";
            case hipErrorFileNotFound:
                return "hipErrorFileNotFound";
            case hipErrorSharedObjectSymbolNotFound:
                return "hipErrorSharedObjectSymbolNotFound";
            case hipErrorSharedObjectInitFailed:
                return "hipErrorSharedObjectInitFailed";
            case hipErrorOperatingSystem:
                return "hipErrorOperatingSystem";
            case hipErrorInvalidHandle:
                return "hipErrorInvalidHandle";
            //case hipErrorInvalidResourceHandle: return "hipErrorInvalidResourceHandle";
            case hipErrorNotFound:
                return "hipErrorNotFound";
            case hipErrorNotReady:
                return "hipErrorNotReady";
            case hipErrorIllegalAddress:
                return "hipErrorIllegalAddress";
            case hipErrorLaunchOutOfResources:
                return "hipErrorLaunchOutOfResources";
            case hipErrorLaunchTimeOut:
                return "hipErrorLaunchTimeOut";
            case hipErrorPeerAccessAlreadyEnabled:
                return "hipErrorPeerAccessAlreadyEnabled";
            case hipErrorPeerAccessNotEnabled:
                return "hipErrorPeerAccessNotEnabled";
            case hipErrorSetOnActiveProcess:
                return "hipErrorSetOnActiveProcess";
            case hipErrorAssert:
                return "hipErrorAssert";
            case hipErrorHostMemoryAlreadyRegistered:
                return "hipErrorHostMemoryAlreadyRegistered";
            case hipErrorHostMemoryNotRegistered:
                return "hipErrorHostMemoryNotRegistered";
            case hipErrorLaunchFailure:
                return "hipErrorLaunchFailure";
            case hipErrorCooperativeLaunchTooLarge:
                return "hipErrorCooperativeLaunchTooLarge";
            case hipErrorNotSupported:
                return "hipErrorNotSupported";
            case hipErrorUnknown:
                return "hipErrorUnknown";
            case hipErrorRuntimeMemory:
                return "hipErrorRuntimeMemory";
            case hipErrorRuntimeOther:
                return "hipErrorRuntimeOther";
            case hipErrorTbd:
                return "hipErrorTbd";
            default:
                return "hipErrorUnknown";
            }
        }

        inline
        hipError_t unlock_memory(void* p)
        {
            if (!p) return hipErrorInvalidValue;

            return hipSuccess;
        }

        inline
        hipError_t wait(Event* e)
        {
            if (!e) return hipErrorInvalidHandle;
            if (!is_done(*e).valid()) return hipErrorInvalidHandle;

            if (is_all_synchronising(*e)) synchronize_device();
            is_done(*e).wait();

            return hipSuccess;
        }
    } // Namespace hip::detail.
} // Namespace hip.
