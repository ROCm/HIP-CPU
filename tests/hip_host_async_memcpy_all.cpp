/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <type_traits>

using namespace std;

constexpr auto p_streams{2};
constexpr auto N{4 * 1024 * 1024};

void simpleNegTest()
{
    float* A_malloc{};

    size_t Nbytes = N * sizeof(float);
    A_malloc = (float*)malloc(Nbytes);

    float* A_pinned{};
    REQUIRE(hipHostMalloc(
        (void**)&A_pinned, Nbytes, hipHostMallocDefault) == hipSuccess);

    float* A_d{};
    REQUIRE(hipMalloc((void**)&A_d, Nbytes) == hipSuccess);

    REQUIRE(hipMemcpyAsync(
        A_pinned, A_d, Nbytes, hipMemcpyDefault, nullptr) == hipSuccess);

    REQUIRE(hipMemcpyAsync(
        A_malloc, A_d, Nbytes, hipMemcpyDeviceToHost, nullptr) == hipSuccess);
}

class Pinned;
class Unpinned;

template<typename T>
struct HostTraits;

template<>
struct HostTraits<Pinned> {
    static const char* Name() { return "Pinned"; };

    static void* Alloc(size_t sizeBytes)
    {
        void* p;
        REQUIRE(hipHostMalloc(
            (void**)&p, sizeBytes, hipHostMallocDefault) == hipSuccess);

        return p;
    }
};

template<typename T>
__global__
void addK(T* A, T K, size_t numElements)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < numElements; i += stride) {
        A[i] = A[i] + K;
    }
}

inline
unsigned int setNumBlocks(
    unsigned int blocksPerCU, unsigned int threadsPerBlock, size_t N)
{
    int device{};
    hipGetDevice(&device);
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, device);

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if (blocks * threadsPerBlock > N) {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    return blocks;
}

constexpr auto blocksPerCU{6};  // to hide latency
constexpr auto threadsPerBlock{256};

//---
// Tests propert dependency resolution between H2D and D2H commands in same stream:
// IN: numInflight : number of copies inflight at any time:
// IN: numPongs = number of iterations to run (iteration)
template<typename T, class AllocType>
void test_pingpong(
    hipStream_t stream,
    size_t numElements,
    int numInflight,
    int numPongs,
    bool doHostSide)
{
    REQUIRE(numElements % numInflight == 0);  // Must be evenly divisible.

    size_t Nbytes = numElements * sizeof(T);
    size_t eachCopyElements = numElements / numInflight;
    size_t eachCopyBytes = eachCopyElements * sizeof(T);

    unsigned blocks = setNumBlocks(blocksPerCU, threadsPerBlock, numElements);

    T* A_h{};

    A_h = (T*)(HostTraits<AllocType>::Alloc(Nbytes));

    T* A_d{};
    REQUIRE(hipMalloc((void**)&A_d, Nbytes) == hipSuccess);

    // Initialize the host array:
    constexpr T initValue = 13;
    constexpr T deviceConst = 2;
    constexpr T hostConst = 10000;
    for (auto i = 0u; i != numElements; ++i) {
        A_h[i] = initValue + i;
    }

    for (int k = 0; k < numPongs; k++) {
        for (int i = 0; i < numInflight; i++) {
            REQUIRE(A_d + i * eachCopyElements < A_d + Nbytes);
            REQUIRE(hipMemcpyAsync(
                &A_d[i * eachCopyElements],
                &A_h[i * eachCopyElements],
                eachCopyBytes,
                hipMemcpyHostToDevice,
                stream) == hipSuccess);
        }

        hipLaunchKernelGGL(
            addK<T>,
            dim3(blocks),
            dim3(threadsPerBlock),
            0,
            stream,
            A_d,
            2,
            numElements);

        for (int i = 0; i < numInflight; i++) {
            REQUIRE(A_d + i * eachCopyElements < A_d + Nbytes);
            REQUIRE(hipMemcpyAsync(
                &A_h[i * eachCopyElements],
                &A_d[i * eachCopyElements],
                eachCopyBytes,
                hipMemcpyDeviceToHost,
                stream) == hipSuccess);
        }

        if (doHostSide) {
            REQUIRE(hipDeviceSynchronize() == hipSuccess);
            for (size_t i = 0; i < numElements; i++) {
                A_h[i] += hostConst;
            }
        }
    };

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    // Verify we copied back all the data correctly:
    for (size_t i = 0; i < numElements; i++) {
        T gold = static_cast<T>(initValue + i);
        // Perform calcs in same order as test above to replicate FP order-of-operations:
        for (int k = 0; k < numPongs; k++) {
            gold += deviceConst;
            if (doHostSide) {
                gold += hostConst;
            }
        }

        if constexpr (is_floating_point_v<T>) REQUIRE(gold == Approx{A_h[i]});
        else REQUIRE(gold == A_h[i]);
    }

    REQUIRE(hipHostFree(A_h) == hipSuccess);
    REQUIRE(hipFree(A_d) == hipSuccess);
}

//---
// Send many async copies to the same stream.
// This requires runtime to keep track of many outstanding commands, and in the case of HCC requires
// growing/tracking the signal pool:
template<typename T>
void test_manyInflightCopies(
    hipStream_t stream, int numElements, int numCopies, bool syncBetweenCopies)
{
    size_t Nbytes = numElements * sizeof(T);
    size_t eachCopyElements = numElements / numCopies;
    size_t eachCopyBytes = eachCopyElements * sizeof(T);

    T* A_h1;
    T* A_h2;

    REQUIRE(hipHostMalloc(
        (void**)&A_h1, Nbytes, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hipHostMalloc(
        (void**)&A_h2, Nbytes, hipHostMallocDefault) == hipSuccess);

    T* A_d;
    REQUIRE(hipMalloc((void**)&A_d, Nbytes) == hipSuccess);

    for (int i = 0; i < numElements; i++) {
        A_h1[i] = 3.14f + static_cast<T>(i);
    }

    for (int i = 0; i < numCopies; i++) {
        REQUIRE(A_d + i * eachCopyElements < A_d + Nbytes);
        REQUIRE(hipMemcpyAsync(
            &A_d[i * eachCopyElements],
            &A_h1[i * eachCopyElements],
            eachCopyBytes,
            hipMemcpyHostToDevice,
            stream) == hipSuccess);
    }

    if (syncBetweenCopies) {
        REQUIRE(hipDeviceSynchronize() == hipSuccess);
    }

    for (int i = 0; i < numCopies; i++) {
        REQUIRE(A_d + i * eachCopyElements < A_d + Nbytes);
        REQUIRE(hipMemcpyAsync(
            &A_h2[i * eachCopyElements],
            &A_d[i * eachCopyElements],
            eachCopyBytes,
            hipMemcpyDeviceToHost,
            stream) == hipSuccess);
    }

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    // Verify we copied back all the data correctly:
    REQUIRE(equal(A_h1, A_h1 + numElements, A_h2, [](auto&& x, auto&& y) {
        if constexpr (is_floating_point_v<T>) return x == Approx{y};
        else return x == y;
    }));

    REQUIRE(hipHostFree(A_h1) == hipSuccess);
    REQUIRE(hipHostFree(A_h2) == hipSuccess);
    REQUIRE(hipFree(A_d) == hipSuccess);
}

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

//---
// Classic example showing how to overlap data transfer with compute.
// We divide the work into "chunks" and create a stream for each chunk.
// Each chunk then runs a H2D copy, followed by kernel execution, followed by D2H copyback.
// Work in separate streams is independent which enables concurrency.

// IN: nStreams : number of streams to use for the test
// IN :useNullStream - use NULL stream.  Synchronizes everything.
// IN: useSyncMemcpyH2D - use sync memcpy (no overlap) for H2D
// IN: useSyncMemcpyD2H - use sync memcpy (no overlap) for D2H
void test_chunkedAsyncExample(
    int nStreams,
    bool useNullStream,
    bool useSyncMemcpyH2D,
    bool useSyncMemcpyD2H)
{
    size_t Nbytes = N * sizeof(int);

    int* A_d;
    int* B_d;
    int* C_d;
    int* A_h;
    int* B_h;
    int* C_h;

    initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, true);

    unsigned blocks = setNumBlocks(blocksPerCU, threadsPerBlock, N);

    vector<hipStream_t> stream(useNullStream ? 1 : nStreams, nullptr);
    if (size(stream) > 1) {
        for (auto&& x : stream) REQUIRE(hipStreamCreate(&x) == hipSuccess);
    }

    size_t workLeft = N;
    size_t workPerStream = N / size(stream);
    for (int i = 0; i < size(stream); ++i) {
        size_t work = (workLeft < workPerStream) ? workLeft : workPerStream;
        size_t workBytes = work * sizeof(int);

        size_t offset = i * workPerStream;

        REQUIRE(A_d + offset < A_d + Nbytes);
        REQUIRE(B_d + offset < B_d + Nbytes);
        REQUIRE(C_d + offset < C_d + Nbytes);

        if (useSyncMemcpyH2D) {
            REQUIRE(hipMemcpy(
                &A_d[offset],
                &A_h[offset],
                workBytes,
                hipMemcpyHostToDevice) == hipSuccess);
            REQUIRE(hipMemcpy(
                &B_d[offset],
                &B_h[offset],
                workBytes,
                hipMemcpyHostToDevice) == hipSuccess);
        }
        else {
            REQUIRE(hipMemcpyAsync(
                &A_d[offset],
                &A_h[offset],
                workBytes,
                hipMemcpyHostToDevice,
                stream[i]) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                &B_d[offset],
                &B_h[offset],
                workBytes,
                hipMemcpyHostToDevice,
                stream[i]) == hipSuccess);
        }

        hipLaunchKernelGGL(
            vectorADD,
            dim3(blocks),
            dim3(threadsPerBlock),
            0,
            stream[i],
            &A_d[offset],
            &B_d[offset],
            &C_d[offset],
            work);

        if (useSyncMemcpyD2H) {
            REQUIRE(hipMemcpy(
                &C_h[offset],
                &C_d[offset],
                workBytes,
                hipMemcpyDeviceToHost) == hipSuccess);
        }
        else {
            REQUIRE(hipMemcpyAsync(
                &C_h[offset],
                &C_d[offset],
                workBytes,
                hipMemcpyDeviceToHost,
                stream[i]) == hipSuccess);
        }
    }

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    for (auto i = 0u; i != N; ++i) REQUIRE(A_h[i] + B_h[i] == C_h[i]);

    freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, true);
}

TEST_CASE("Async memcpy", "[host][memcpy][async]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);

    simpleNegTest();
}

TEST_CASE("Async memcpy, many in-flight copies", "[host][memcpy][async]")
{
    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    test_manyInflightCopies<float>(stream, 1024, 16, true);
    test_manyInflightCopies<float>(stream, 1024, 4, true);
    test_manyInflightCopies<float>(stream, 1024 * 8, 64, false);

    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE("Async memcpy, chunked", "[host][memcpy][async]")
{
    test_chunkedAsyncExample(p_streams, true, true, true);    // Easy sync version
    test_chunkedAsyncExample(p_streams, false, true, true);   // Easy sync version
    test_chunkedAsyncExample(p_streams, false, false, true);  // Some async
    test_chunkedAsyncExample(p_streams, false, false, false); // All async
}

TEST_CASE("Async memcpy, pingpong", "[host][memcpy][async]")
{
    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    test_pingpong<int, Pinned>(stream, 1024*1024*32, 1, 1, false);
    test_pingpong<int, Pinned>(stream, 1024*1024*32, 1, 10, false);

    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}