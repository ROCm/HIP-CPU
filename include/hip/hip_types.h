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

#include "../../src/include/hip/detail/context.hpp"
#include "../../src/include/hip/detail/device.hpp"
#include "../../src/include/hip/detail/event.hpp"
#include "../../src/include/hip/detail/function.hpp"
#include "../../src/include/hip/detail/module.hpp"
#include "../../src/include/hip/detail/stream.hpp"
#include "../../src/include/hip/detail/types.hpp"

#include <cstddef>
#include <cstdint>

using dim3 = hip::detail::Dim3;
using hipCtx_t = hip::detail::Context*;
using hipDevice_t = hip::detail::Device; // TODO

// BEGIN STRUCT HIPDEVICEARCH_T
struct hipDeviceArch_t final {
    // 32-bit Atomics
    std::uint32_t hasGlobalInt32Atomics    : 1; // 32-bit integer atomics for global memory.
    std::uint32_t hasGlobalFloatAtomicExch : 1; // 32-bit float atomic exch for global memory.
    std::uint32_t hasSharedInt32Atomics    : 1; // 32-bit integer atomics for shared memory.
    std::uint32_t hasSharedFloatAtomicExch : 1; // 32-bit float atomic exch for shared memory.
    std::uint32_t hasFloatAtomicAdd        : 1; // 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    std::uint32_t hasGlobalInt64Atomics : 1; // 64-bit integer atomics for global memory.
    std::uint32_t hasSharedInt64Atomics : 1; // 64-bit integer atomics for shared memory.

    // Doubles
    std::uint32_t hasDoubles : 1; // Double-precision floating point.

    // Warp cross-lane operations
    std::uint32_t hasWarpVote    : 1; // Warp vote instructions (__any, __all).
    std::uint32_t hasWarpBallot  : 1; // Warp ballot instructions (__ballot).
    std::uint32_t hasWarpShuffle : 1; // Warp shuffle operations. (__shfl_*).
    std::uint32_t hasFunnelShift : 1; // Funnel two words into one with shift&mask caps.

    // Sync
    std::uint32_t hasThreadFenceSystem : 1; // __threadfence_system.
    std::uint32_t hasSyncThreadsExt    : 1; // __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    std::uint32_t hasSurfaceFuncs       : 1; // Surface functions.
    std::uint32_t has3dGrid             : 1; // Grid and group dims are 3D (rather than 2D).
    std::uint32_t hasDynamicParallelism : 1; // Dynamic parallelism.
};
// END STRUCT HIPDEVICEARCH_T

// BEGIN STRUCT HIPDEVICEPROP_T
struct hipDeviceProp_t final {
    char name[256];                                        // Device name.
    std::size_t totalGlobalMem;                            // Size of global memory region (in bytes).
    std::size_t sharedMemPerBlock;                         // Size of shared memory region (in bytes).
    std::int32_t regsPerBlock;                             // Registers per block.
    std::int32_t warpSize;                                 // Warp size.
    std::int32_t maxThreadsPerBlock;                       // Max work items per work group or workgroup max size.
    std::int32_t maxThreadsDim[3];                         // Max number of threads in each dimension (XYZ) of a block.
    std::int32_t maxGridSize[3];                           // Max grid dimensions (XYZ).
    std::int32_t clockRate;                                // Max clock frequency of the multiProcessors in khz.
    std::int32_t memoryClockRate;                          // Max global memory clock frequency in khz.
    std::int32_t memoryBusWidth;                           // Global memory bus width in bits.
    std::size_t totalConstMem;                             // Size of shared memory region (in bytes).
    std::int32_t major;                                    // Major compute capability. On HCC, this is an approximation and features may
                                                           // differ from CUDA CC. See the arch feature flags for portable ways to query
                                                           // feature caps.
    std::int32_t minor;                                    // Minor compute capability. On HCC, this is an approximation and features may
                                                           // differ from CUDA CC.  See the arch feature flags for portable ways to query
                                                           // feature caps.
    std::int32_t multiProcessorCount;                      // Number of multi-processors (compute units).
    std::int32_t l2CacheSize;                              // L2 cache size.
    std::int32_t maxThreadsPerMultiProcessor;              // Maximum resident threads per multi-processor.
    std::int32_t computeMode;                              // Compute mode.
    std::int32_t clockInstructionRate;                     // Frequency in khz of the timer used by the device-side "clock*"
                                                           // instructions. New for HIP.
    hipDeviceArch_t arch;                                  // Architectural feature flags. New for HIP.
    std::int32_t concurrentKernels;                        // Device can possibly execute multiple kernels concurrently.
    std::int32_t pciDomainID;                              // PCI Domain ID
    std::int32_t pciBusID;                                 // PCI Bus ID.
    std::int32_t pciDeviceID;                              // PCI Device ID.
    std::size_t maxSharedMemoryPerMultiProcessor;          // Maximum Shared Memory Per Multiprocessor.
    std::int32_t isMultiGpuBoard;                          // 1 if device is on a multi-GPU board, 0 if not.
    std::int32_t canMapHostMemory;                         // Check whether HIP can map host memory
    std::int32_t gcnArch;                                  // AMD GCN Arch Value. Eg: 803, 701
    std::int32_t integrated;                               // APU vs dGPU
    std::int32_t cooperativeLaunch;                        // HIP device supports cooperative launch
    std::int32_t cooperativeMultiDeviceLaunch;             // HIP device supports cooperative launch on multiple devices
    std::int32_t maxTexture1D;                             // Maximum number of elements in 1D images
    std::int32_t maxTexture2D[2];                          // Maximum dimensions (width, height) of 2D images, in image elements
    std::int32_t maxTexture3D[3];                          // Maximum dimensions (width, height, depth) of 3D images, in image elements
    std::uint32_t* hdpMemFlushCntl;                        // Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    std::uint32_t* hdpRegFlushCntl;                        // Address of HDP_REG_COHERENCY_FLUSH_CNTL register
    std::size_t memPitch;                                  // Maximum pitch in bytes allowed by memory copies
    std::size_t textureAlignment;                          // Alignment requirement for textures
    std::size_t texturePitchAlignment;                     // Pitch alignment requirement for texture references bound to pitched memory
    std::int32_t kernelExecTimeoutEnabled;                 // Run time limit for kernels executed on the device
    std::int32_t ECCEnabled;                               // Device has ECC support enabled
    std::int32_t tccDriver;                                // 1:If device is Tesla device using TCC driver, else 0
    std::int32_t cooperativeMultiDeviceUnmatchedFunc;      // HIP device supports cooperative launch on multiple
                                                           // devices with unmatched functions
    std::int32_t cooperativeMultiDeviceUnmatchedGridDim;   // HIP device supports cooperative launch on multiple
                                                           // devices with unmatched grid dimensions
    std::int32_t cooperativeMultiDeviceUnmatchedBlockDim;  // HIP device supports cooperative launch on multiple
                                                           // devices with unmatched block dimensions
    std::int32_t cooperativeMultiDeviceUnmatchedSharedMem; // HIP device supports cooperative launch on multiple
                                                           // devices with unmatched shared memories
    std::int32_t deviceOverlap;                            // HIP device can concurrently copy memory and execute a kernel
};
// END STRUCT HIPDEVICEPROP_T

using hipDeviceptr_t = void*;
using hipEvent_t = hip::detail::Event*;
using hipFunction_t = hip::detail::Function*;

// BEGIN STRUCT HIPFUNCATTRIBUTES
struct hipFuncAttributes final {
    std::int32_t binaryVersion{9999};
    std::size_t constSizeBytes;
    std::size_t localSizeBytes;
    std::int32_t maxThreadsPerBlock;
    std::int32_t numRegs;
    std::int32_t ptxVersion{9999};
    std::size_t sharedSizeBytes;
};
// END STRUCT HIPFUNCATTRIBUTES

using hipModule_t = hip::detail::Module*;
using hipStream_t = hip::detail::Stream*;