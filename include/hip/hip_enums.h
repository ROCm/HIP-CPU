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

enum hipDeviceFlags_t {
    hipDeviceScheduleAuto         = 0, // Automatically select between Spin and Yield
    hipDeviceScheduleSpin         = 1, // Dedicate a CPU core to spin-wait. Provides lowest latency, but burns
                                       // a CPU core and may consume more power.
    hipDeviceScheduleYield        = 2, // Yield the CPU to the operating system when waiting. May increase
                                       // latency, but lowers power and is friendlier to other threads in the
                                       // system.
    hipDeviceScheduleBlockingSync = 4,
    hipDeviceMapHost              = 8,
    hipDeviceLmemResizeToMax      = 16
};

enum hipError_t {
    hipSuccess                          = 0,    // Successful completion.
    hipErrorInvalidValue                = 1,    // One or more of the parameters passed to the API call is NULL
                                                // or not in an acceptable range.
    hipErrorOutOfMemory                 = 2,
    // Deprecated
    hipErrorMemoryAllocation            = 2,    // Memory allocation error.
    hipErrorNotInitialized              = 3,
    // Deprecated
    hipErrorInitializationError         = 3,
    hipErrorDeinitialized               = 4,
    hipErrorProfilerDisabled            = 5,
    hipErrorProfilerNotInitialized      = 6,
    hipErrorProfilerAlreadyStarted      = 7,
    hipErrorProfilerAlreadyStopped      = 8,
    hipErrorInvalidConfiguration        = 9,
    hipErrorInvalidSymbol               = 13,
    hipErrorInvalidDevicePointer        = 17,   // Invalid Device Pointer
    hipErrorInvalidMemcpyDirection      = 21,   // Invalid memory copy direction
    hipErrorInsufficientDriver          = 35,
    hipErrorMissingConfiguration        = 52,
    hipErrorPriorLaunchFailure          = 53,
    hipErrorInvalidDeviceFunction       = 98,
    hipErrorNoDevice                    = 100,  // Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice               = 101,  // DeviceID must be in range 0...#compute-devices.
    hipErrorInvalidImage                = 200,
    hipErrorInvalidContext              = 201,  // Produced when input context is invalid.
    hipErrorContextAlreadyCurrent       = 202,
    hipErrorMapFailed                   = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed       = 205,  // Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed                 = 206,
    hipErrorArrayIsMapped               = 207,
    hipErrorAlreadyMapped               = 208,
    hipErrorNoBinaryForGpu              = 209,
    hipErrorAlreadyAcquired             = 210,
    hipErrorNotMapped                   = 211,
    hipErrorNotMappedAsArray            = 212,
    hipErrorNotMappedAsPointer          = 213,
    hipErrorECCNotCorrectable           = 214,
    hipErrorUnsupportedLimit            = 215,
    hipErrorContextAlreadyInUse         = 216,
    hipErrorPeerAccessUnsupported       = 217,
    hipErrorInvalidKernelFile           = 218,  // In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext      = 219,
    hipErrorInvalidSource               = 300,
    hipErrorFileNotFound                = 301,
    hipErrorSharedObjectSymbolNotFound  = 302,
    hipErrorSharedObjectInitFailed      = 303,
    hipErrorOperatingSystem             = 304,
    hipErrorInvalidHandle               = 400,
    // Deprecated
    hipErrorInvalidResourceHandle       = 400,  // Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorNotFound                    = 500,
    hipErrorNotReady                    = 600,  // Indicates that asynchronous operations enqueued earlier are not
                                                // ready.  This is not actually an error, but is used to distinguish
                                                // from hipSuccess (which indicates completion).  APIs that return
                                                // this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress              = 700,
    hipErrorLaunchOutOfResources        = 701,  // Out of resources error.
    hipErrorLaunchTimeOut               = 702,
    hipErrorPeerAccessAlreadyEnabled    = 704,  // Peer access was already enabled from the current device.
    hipErrorPeerAccessNotEnabled        = 705,  // Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess          = 708,
    hipErrorAssert                      = 710,  // Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,  // Produced when trying to lock a page-locked memory.
    hipErrorHostMemoryNotRegistered     = 713,  // Produced when trying to unlock a non-page-locked memory.
    hipErrorLaunchFailure               = 719,  // An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge   = 720,  // This error indicates that the number of blocks launched per grid for a kernel
                                                // that was launched via cooperative launch APIs exceeds the maximum number of
                                                // allowed blocks for the current device
    hipErrorNotSupported                = 801,  // Produced when the hip API is not supported/implemented
    hipErrorUnknown                     = 999,  // Unknown error.
                                                // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory               = 1052, // HSA runtime memory call returned error.  Typically not seen
                                                // in production systems.
    hipErrorRuntimeOther                = 1053, // HSA runtime call other than memory returned error.  Typically
                                                // not seen in production systems.
    hipErrorTbd                                 // Marker that more error codes are needed.
};

enum hipEventFlags {
    hipEventDefault         = 0,          // Default flags
    hipEventBlockingSync    = 1,          // Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.
    hipEventDisableTiming   = 0x2,        // Disable event's capability to record timing information. May improve performance.
    hipEventInterprocess    = 0x4,        // Event can support IPC.
    hipEventReleaseToDevice = 0x40000000, // Use a device-scope release when recording this event. This flag is useful to obtain
                                          // more precise timings of commands between events. The flag is a no-op.
    hipEventReleaseToSystem = 0x80000000  // Use a system-scope release that when recording this event. This flag is useful to
                                          // make non-coherent host memory visible to the host. The flag is a no-op.
};

enum hipHostMallocKind {
    hipHostMallocDefault       = 0,
    hipHostMallocPortable      = 1,          // Memory is considered allocated by all contexts.
    hipHostMallocMapped        = 2,          // Map the allocation into the address space for the current device.
                                             // The device pointer can be obtained with hipHostGetDevicePointer.
    hipHostMallocWriteCombined = 4,
    hipHostMallocCoherent      = 0x40000000, // Allocate coherent memory. Overrides HIP_COHERENT_HOST_ALLOC for
                                             // specific allocation.
    hipHostMallocNonCoherent   = 0x80000000  // Allocate non-coherent memory. Overrides HIP_COHERENT_HOST_ALLOC
                                             // for specific allocation.
};

enum hipLimit_t {
    hipLimitMallocHeapSize = 0x02
};

enum hipMallocManagedFlags {
    hipMemAttachGlobal = 0,
    hipMemAttachHost   = 1
};

enum hipMemcpyKind {
    hipMemcpyHostToHost     = 0,
    hipMemcpyHostToDevice   = 1,
    hipMemcpyDeviceToHost   = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault        = 4
};

enum hipRegisterKind {
    hipHostRegisterDefault          = 0, // Memory is Mapped and Portable
    hipHostRegisterPortable         = 1, // Memory is considered registered by all contexts.
    hipHostRegisterMapped           = 2, // Map the allocation into the address space for the current device. The device pointer
                                         // can be obtained with #hipHostGetDevicePointer.
    hipHostRegisterIoMemory         = 4, // Not supported.
    hipExtHostRegisterCoarseGrained = 8  // Coarse Grained host memory lock
};

enum hipStreamFlags {
    hipStreamDefault = 0,    // Default stream creation flags. These are used with hipStreamCreate().
    hipStreamNonBlocking = 1 // Stream does not implicitly synchronize with null stream
};