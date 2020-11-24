/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdint>
#include <cstdlib>
#include <thread>

using namespace std;

template<typename T>
__global__
void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t n)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < n; i += stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}

template <typename T>
class DeviceMemory {
    T* _A_d;
    T* _B_d;
    T* _C_d;
    T* _C_dd;

    size_t _maxNumElements;
    int _offset;
public:
    DeviceMemory(size_t numElements);
    ~DeviceMemory();

    T* A_d() const { return _A_d + _offset; };
    T* B_d() const { return _B_d + _offset; };
    T* C_d() const { return _C_d + _offset; };
    T* C_dd() const { return _C_dd + _offset; };

    size_t maxNumElements() const { return _maxNumElements; };

    void offset(int offset) { _offset = offset; };
    int offset() const { return _offset; };
};

template<typename T>
inline
void setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h)
{
    // Initialize the host data:
    for (size_t i = 0; i < numElements; i++) {
        if (A_h) A_h[i] = static_cast<T>(3.146 + i);  // Pi
        if (B_h) B_h[i] = static_cast<T>(1.618 + i);  // Phi
        if (C_h) C_h[i] = static_cast<T>(0.0 + i);
    }
}

template<typename T>
inline
void initArraysForHost(
    T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false) {
    size_t Nbytes = N * sizeof(T);

    if (usePinnedHost) {
        if (A_h) REQUIRE(hipHostMalloc((void**)A_h, Nbytes) == hipSuccess);
        if (B_h) REQUIRE(hipHostMalloc((void**)B_h, Nbytes) == hipSuccess);
        if (C_h) REQUIRE(hipHostMalloc((void**)C_h, Nbytes) == hipSuccess);
    }
    else {
        if (A_h) {
            *A_h = (T*)malloc(Nbytes);
            REQUIRE(*A_h != nullptr);
        }

        if (B_h) {
            *B_h = (T*)malloc(Nbytes);
            REQUIRE(*B_h != nullptr);
        }

        if (C_h) {
            *C_h = (T*)malloc(Nbytes);
            REQUIRE(*C_h != nullptr);
        }
    }

    setDefaultData(N, A_h ? *A_h : NULL, B_h ? *B_h : NULL, C_h ? *C_h : NULL);
}

template<typename T>
inline
void initArrays(
    T** A_d,
    T** B_d,
    T** C_d,
    T** A_h,
    T** B_h,
    T** C_h,
    size_t N,
    bool usePinnedHost = false)
{
    size_t Nbytes = N * sizeof(T);

    if (A_d) REQUIRE(hipMalloc((void**)A_d, Nbytes) == hipSuccess);
    if (B_d) REQUIRE(hipMalloc((void**)B_d, Nbytes) == hipSuccess);
    if (C_d) REQUIRE(hipMalloc((void**)C_d, Nbytes) == hipSuccess);

    initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}

template<typename T>
inline
DeviceMemory<T>::DeviceMemory(size_t numElements)
    : _maxNumElements(numElements), _offset(0)
{
    T** np = nullptr;
    initArrays(&_A_d, &_B_d, &_C_d, np, np, np, numElements, 0);

    size_t sizeElements = numElements * sizeof(T);

    REQUIRE(hipMalloc((void**)&_C_dd, sizeElements) == hipSuccess);
}

template<typename T>
inline
void freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost)
{
    if (usePinnedHost) {
        if (A_h) REQUIRE(hipHostFree(A_h) == hipSuccess);
        if (B_h) REQUIRE(hipHostFree(B_h) == hipSuccess);
        if (C_h) REQUIRE(hipHostFree(C_h) == hipSuccess);
    }
    else {
        if (A_h) free(A_h);
        if (B_h) free(B_h);
        if (C_h) free(C_h);
    }
}

template<typename T>
inline
void freeArrays(
    T* A_d, T* B_d, T* C_d, T* A_h, T* B_h, T* C_h, bool usePinnedHost)
{
    if (A_d) REQUIRE(hipFree(A_d) == hipSuccess);
    if (B_d) REQUIRE(hipFree(B_d) == hipSuccess);
    if (C_d) REQUIRE(hipFree(C_d) == hipSuccess);

    freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}

template<typename T>
inline
DeviceMemory<T>::~DeviceMemory()
{
    T* np = nullptr;
    freeArrays(_A_d, _B_d, _C_d, np, np, np, 0);

    REQUIRE(hipFree(_C_dd) == hipSuccess);

    _C_dd = nullptr;
}


template<typename T>
class HostMemory {
    size_t _maxNumElements;

    int _offset;

    // Host arrays
    T* _A_h;
    T* _B_h;
    T* _C_h;
public:
    HostMemory(size_t numElements, bool usePinnedHost);
    void reset(size_t numElements, bool full = false);
    ~HostMemory();

    T* A_h() const { return _A_h + _offset; };
    T* B_h() const { return _B_h + _offset; };
    T* C_h() const { return _C_h + _offset; };

    size_t maxNumElements() const { return _maxNumElements; };

    void offset(int offset) { _offset = offset; };
    int offset() const { return _offset; };

    // Host arrays, secondary copy
    T* A_hh;
    T* B_hh;

    bool _usePinnedHost;
};

template<typename T>
inline
HostMemory<T>::HostMemory(size_t numElements, bool usePinnedHost)
    : _maxNumElements(numElements), _usePinnedHost(usePinnedHost), _offset(0)
{
    T** np = nullptr;
    initArrays(np, np, np, &_A_h, &_B_h, &_C_h, numElements, usePinnedHost);

    A_hh = nullptr;
    B_hh = nullptr;

    size_t sizeElements = numElements * sizeof(T);

    if (usePinnedHost) {
        REQUIRE(hipHostMalloc(
            (void**)&A_hh, sizeElements, hipHostMallocDefault) == hipSuccess);
        REQUIRE(hipHostMalloc(
            (void**)&B_hh, sizeElements, hipHostMallocDefault) == hipSuccess);
    } else {
        A_hh = (T*)malloc(sizeElements);
        B_hh = (T*)malloc(sizeElements);
    }
}

template<typename T>
inline
void HostMemory<T>::reset(size_t numElements, bool full)
{
    // Initialize the host data:
    for (size_t i = 0; i < numElements; i++) {
        A_hh[i] = static_cast<T>(1097.0 + i);
        B_hh[i] = static_cast<T>(1492.0 + i);  // Phi

        if (full) {
            _A_h[i] = static_cast<T>(3.146 + i);  // Pi
            _B_h[i] = static_cast<T>(1.618 + i);  // Phi
        }
    }
}

template<typename T>
inline
HostMemory<T>::~HostMemory()
{
    freeArraysForHost(_A_h, _B_h, _C_h, _usePinnedHost);

    if (_usePinnedHost) {
        REQUIRE(hipHostFree(A_hh) == hipSuccess);
        REQUIRE(hipHostFree(B_hh) == hipSuccess);

    } else {
        free(A_hh);
        free(B_hh);
    }
};

inline
unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N)
{
    int device;
    REQUIRE(hipGetDevice(&device) == hipSuccess);

    hipDeviceProp_t props;
    REQUIRE(hipGetDeviceProperties(&props, device) == hipSuccess);

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if (blocks * threadsPerBlock > N) {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    return blocks;
}

constexpr auto blocksPerCU{6};  // to hide latency
constexpr auto threadsPerBlock{256};

//---
// Test many different kinds of memory copies.
// The subroutine allocates memory , copies to device, runs a vector add kernel, copies back, and
// checks the result.
//
// IN: numElements  controls the number of elements used for allocations.
// IN: usePinnedHost : If true, allocate host with hipHostMalloc and is pinned ; else allocate host
// memory with malloc. IN: useHostToHost : If true, add an extra host-to-host copy. IN:
// useDeviceToDevice : If true, add an extra deviceto-device copy after result is produced. IN:
// useMemkindDefault : If true, use memkinddefault (runtime figures out direction).  if false, use
// explicit memcpy direction.
//
template<typename T>
inline
void memcpytest2(
    DeviceMemory<T>* dmem,
    HostMemory<T>* hmem,
    size_t numElements,
    bool useHostToHost,
    bool useDeviceToDevice,
    bool useMemkindDefault)
{
    size_t sizeElements = numElements * sizeof(T);

    hmem->reset(numElements);
    unsigned blocks = setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    REQUIRE(numElements <= dmem->maxNumElements());
    REQUIRE(numElements <= hmem->maxNumElements());

    if (useHostToHost) {
        // Do some extra host-to-host copies here to mix things up:
        REQUIRE(hipMemcpy(
            hmem->A_hh, hmem->A_h(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToHost)
            == hipSuccess);
        REQUIRE(hipMemcpy(
            hmem->B_hh,
            hmem->B_h(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToHost)
            == hipSuccess);

        REQUIRE(hipMemcpy(
            dmem->A_d(),
            hmem->A_hh,
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice)
            == hipSuccess);
        REQUIRE(hipMemcpy(
            dmem->B_d(),
            hmem->B_hh,
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice)
            == hipSuccess);
    }
    else {
        REQUIRE(hipMemcpy(
            dmem->A_d(),
            hmem->A_h(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice)
            == hipSuccess);
        REQUIRE(hipMemcpy(
            dmem->B_d(),
            hmem->B_h(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyHostToDevice)
            == hipSuccess);
    }

    hipLaunchKernelGGL(
        vectorADD,
        dim3(blocks),
        dim3(threadsPerBlock),
        0,
        0,
        dmem->A_d(),
        dmem->B_d(),
        dmem->C_d(),
        numElements);

    if (useDeviceToDevice) {
        // Do an extra device-to-device copy here to mix things up:
        REQUIRE(hipMemcpy(
            dmem->C_dd(),
            dmem->C_d(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToDevice)
            == hipSuccess);

        // Destroy the original dmem->C_d():
        REQUIRE(hipMemset(dmem->C_d(), 0x5A, sizeElements) == hipSuccess);

        REQUIRE(hipMemcpy(
            hmem->C_h(),
            dmem->C_dd(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToHost)
            == hipSuccess);
    }
    else {
        REQUIRE(hipMemcpy(
            hmem->C_h(),
            dmem->C_d(),
            sizeElements,
            useMemkindDefault ? hipMemcpyDefault : hipMemcpyDeviceToHost)
            == hipSuccess);
    }

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    for (auto i = 0u; i != numElements; ++i) {
        if constexpr (is_floating_point_v<T>) {
            REQUIRE(hmem->A_h()[i] + hmem->B_h()[i] == Approx{hmem->C_h()[i]});
        }
        else {
            REQUIRE(static_cast<T>(
                hmem->A_h()[i] + hmem->B_h()[i]) == hmem->C_h()[i]);
        }
    }
}


//---
// Try all the 16 possible combinations to memcpytest2 - usePinnedHost, useHostToHost,
// useDeviceToDevice, useMemkindDefault
template<typename T>
inline
void memcpytest2_for_type(size_t numElements)
{
    DeviceMemory<T> memD(numElements);
    HostMemory<T> memU(numElements, 0 /*usePinnedHost*/);
    HostMemory<T> memP(numElements, 1 /*usePinnedHost*/);

    for (int usePinnedHost = 0; usePinnedHost <= 1; usePinnedHost++) {
        for (int useHostToHost = 0; useHostToHost <= 1; useHostToHost++) {  // TODO
            for (int useDeviceToDevice = 0; useDeviceToDevice <= 1; useDeviceToDevice++) {
                for (int useMemkindDefault = 0; useMemkindDefault <= 1; useMemkindDefault++) {
                    memcpytest2<T>(&memD, usePinnedHost ? &memP : &memU, numElements, useHostToHost,
                                   useDeviceToDevice, useMemkindDefault);
                }
            }
        }
    }
}

void memcpytest2_get_host_memory(size_t& free, size_t& total) {
    // TODO: remove use of internal details
    const auto mem{hip::detail::System::memory()};

    free = mem.available;
    total = mem.total;
}

//---
// Try many different sizes to memory copy.
template<typename T>
inline
void memcpytest2_sizes(size_t maxElem = 0)
{
    int deviceId;
    REQUIRE(hipGetDevice(&deviceId) == hipSuccess);

    size_t free, total, freeCPU, totalCPU;
    REQUIRE(hipMemGetInfo(&free, &total) == hipSuccess);
    memcpytest2_get_host_memory(freeCPU, totalCPU);

    if (maxElem == 0) {
        // Use lesser maxElem if not enough host memory available
        size_t maxElemGPU = 3 * 1024 * 1024 / sizeof(T);
        size_t maxElemCPU = 3 * 1024 * 1024 / sizeof(T);
        maxElem = maxElemGPU < maxElemCPU ? maxElemGPU : maxElemCPU;
    }

    REQUIRE(hipDeviceReset() == hipSuccess);

    DeviceMemory<T> memD(maxElem);
    HostMemory<T> memU(maxElem, 0 /*usePinnedHost*/);
    HostMemory<T> memP(maxElem, 1 /*usePinnedHost*/);

    for (size_t elem = 1; elem <= maxElem; elem *= 2) {
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }
}


//---
// Try many different sizes to memory copy.
template<typename T>
inline
void memcpytest2_offsets(size_t maxElem, bool devOffsets, bool hostOffsets)
{
    int deviceId;
    REQUIRE(hipGetDevice(&deviceId) == hipSuccess);

    size_t free, total;
    REQUIRE(hipMemGetInfo(&free, &total) == hipSuccess);

    REQUIRE(hipDeviceReset() == hipSuccess);

    DeviceMemory<T> memD(maxElem);
    HostMemory<T> memU(maxElem, 0 /*usePinnedHost*/);
    HostMemory<T> memP(maxElem, 1 /*usePinnedHost*/);

    size_t elem = maxElem / 2;

    for (int offset = 0; offset < 512; offset++) {
        REQUIRE(elem + offset < maxElem);
        if (devOffsets) {
            memD.offset(offset);
        }
        if (hostOffsets) {
            memU.offset(offset);
            memP.offset(offset);
        }
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }

    for (int offset = 512; offset < elem; offset *= 2) {
        REQUIRE(elem + offset < maxElem);
        if (devOffsets) {
            memD.offset(offset);
        }
        if (hostOffsets) {
            memU.offset(offset);
            memP.offset(offset);
        }
        memcpytest2<T>(&memD, &memU, elem, 1, 1, 0);  // unpinned host
        memcpytest2<T>(&memD, &memP, elem, 1, 1, 0);  // pinned host
    }
}

constexpr auto N{4 * 1024 * 1024};

//---
// Create multiple threads to stress multi-thread locking behavior in the
// allocation/deallocation/tracking logic:
template<typename T>
inline
void multiThread_1(bool serialize, bool usePinnedHost)
{
    DeviceMemory<T> memD(N);
    HostMemory<T> mem1(N, usePinnedHost);
    HostMemory<T> mem2(N, usePinnedHost);

    thread t1(memcpytest2<T>, &memD, &mem1, N, 0, 0, 0);
    if (serialize) {
        t1.join();
    }


    thread t2(memcpytest2<T>, &memD, &mem2, N, 0, 0, 0);
    if (serialize) {
        t2.join();
    }

    if (!serialize) {
        t1.join();
        t2.join();
    }
}

TEMPLATE_TEST_CASE(
    "memcpy test types",
    "[host][memcpy][types]",
    float,
    double,
    unsigned char,
    int)
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    REQUIRE(hipDeviceReset() == hipSuccess);

    memcpytest2_for_type<TestType>(N);
    memcpytest2_for_type<TestType>(N);
    memcpytest2_for_type<TestType>(N);
    memcpytest2_for_type<TestType>(N);
}

TEST_CASE("memcpy test sizes 64KiB boundary", "[host][memcpy][sizes][64K]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    REQUIRE(hipDeviceReset() == hipSuccess);

    // Some tests around the 64KB boundary which have historically shown issues:
    size_t maxElem = 32 * 1024 * 1024;
    DeviceMemory<float> memD(maxElem);
    HostMemory<float> memU(maxElem, 0 /*usePinnedHost*/);
    HostMemory<float> memP(maxElem, 0 /*usePinnedHost*/);

    // These all pass:
    memcpytest2<float>(&memD, &memP, 15 * 1024 * 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memP, 16 * 1024 * 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memP, 16 * 1024 * 1024 + 16 * 1024, 0, 0, 0);

    // Just over 64MB:
    memcpytest2<float>(&memD, &memP, 16 * 1024 * 1024 + 512 * 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memP, 17 * 1024 * 1024 + 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memP, 32 * 1024 * 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memU, 32 * 1024 * 1024, 0, 0, 0);
    memcpytest2<float>(&memD, &memP, 32 * 1024 * 1024, 1, 1, 0);
    memcpytest2<float>(&memD, &memP, 32 * 1024 * 1024, 1, 1, 0);
}

TEST_CASE("memcpy test sizes", "[host][memcpy][sizes]")
{
    REQUIRE(hipDeviceReset() == hipSuccess);

    memcpytest2_sizes<float>(0);
}

TEST_CASE("memcpy multithreaded test", "[host][multithread][memcpy]")
{
    REQUIRE(hipDeviceReset() == hipSuccess);

    // Simplest cases: serialize the threads, and also used pinned memory:
    // This verifies that the sub-calls to memcpytest2 are correct.
    multiThread_1<float>(true, true);

    // Serialize, but use unpinned memory to stress the unpinned memory xfer path.
    multiThread_1<float>(true, false);

    // Remove serialization, so two threads are performing memory copies in parallel.
    multiThread_1<float>(false, true);

    // Remove serialization, and use unpinned.
    multiThread_1<float>(false, false);  // TODO
}

TEMPLATE_TEST_CASE(
    "memcpy device offsets",
    "[host][memcpy][offsets]",
    unsigned char,
    float,
    double)
{
    REQUIRE(hipDeviceReset() == hipSuccess);

    size_t maxSize = 256 * 1024;
    memcpytest2_offsets<TestType>(maxSize, true, false);
}