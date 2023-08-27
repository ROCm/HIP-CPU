/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdlib>
#include <vector>
#include <utility>

using namespace std;

constexpr auto blocksPerCU{6u};
constexpr auto p_iters{10u};
constexpr auto N{4000000};
constexpr auto threadsPerBlock{256};

inline
unsigned int set_num_blocks(
    unsigned int blocksPerCU, unsigned int threadsPerBlock, unsigned int N)
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


template<typename T>
__global__
void vectorADDReverse(const T* A_d, const T* B_d, T* C_d, int n)
{
    int offset = (blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;

    for (int i = n - stride + offset; i >= 0; i -= stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}

TEST_CASE("Serial, sync memcpy, same stream.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    vector<float> A(N, 1000.0f);
    vector<float> B(N, 2000.0f);
    vector<float> C(N, -1.f);

    float* A_d{};
    float* B_d{};
    float* C_d{};

    REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

    REQUIRE(hipMemcpy(A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
    REQUIRE(hipMemcpy(B_d, data(B), sizeof(float) * size(B)) == hipSuccess);
    REQUIRE(hipMemcpy(C_d, data(C), sizeof(float) * size(C)) == hipSuccess);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    A.assign(size(A), 1.0f);
    B.assign(size(B), 2.0f);
    C.assign(size(C), -1.f);

    const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
    for (auto i = 0u; i != p_iters; ++i) {
        REQUIRE(hipMemcpy(A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMemcpy(B_d, data(B), sizeof(float) * size(B)) == hipSuccess);

        hipLaunchKernelGGL(
            vectorADDReverse,
            dim3(blocks),
            dim3(threadsPerBlock),
            0,
            nullptr,
            A_d,
            B_d,
            C_d,
            N);

        REQUIRE(hipMemcpy(data(C), C_d, sizeof(float) * size(C)) == hipSuccess);

        for (auto j = 0u; j != size(C); ++j) {
            REQUIRE(Approx{C[j]} == A[j] + B[j]);
        }
        REQUIRE(hipDeviceSynchronize() == hipSuccess);

    }
    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE("Serial, async memcpy, same stream.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    vector<float> A(N, 1000.0f);
    vector<float> B(N, 2000.0f);
    vector<float> C(N, -1.f);

    float* A_d{};
    float* B_d{};
    float* C_d{};

    REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

    REQUIRE(hipMemcpyAsync(
        A_d,
        data(A),
        sizeof(float) * size(A),
        hipMemcpyHostToDevice,
        stream) == hipSuccess);
    REQUIRE(hipMemcpyAsync(
        B_d,
        data(B),
        sizeof(float) * size(B),
        hipMemcpyHostToDevice,
        stream) == hipSuccess);
    REQUIRE(hipMemcpyAsync(
        C_d,
        data(C),
        sizeof(float) * size(C),
        hipMemcpyHostToDevice,
        stream) == hipSuccess);

    REQUIRE(hipDeviceSynchronize() == hipSuccess);

    A.assign(size(A), 1.0f);
    B.assign(size(B), 2.0f);
    C.assign(size(C), -1.f);

    const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
    for (auto i = 0u; i != p_iters; ++i) {
        REQUIRE(hipMemcpyAsync(
            A_d,
            data(A),
            sizeof(float) * size(A),
            hipMemcpyHostToDevice,
            stream) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d,
            data(B),
            sizeof(float) * size(B),
            hipMemcpyHostToDevice,
            stream) == hipSuccess);

        hipLaunchKernelGGL(
            vectorADDReverse,
            dim3(blocks),
            dim3(threadsPerBlock),
            0,
            stream,
            A_d,
            B_d,
            C_d,
            N);

        REQUIRE(hipMemcpyAsync(
            data(C),
            C_d,
            sizeof(float) * size(C),
            hipMemcpyHostToDevice,
            stream) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        for (auto j = 0u; j != size(C); ++j) {
            REQUIRE(Approx{C[j]} == A[j] + B[j]);
        }
    }

    REQUIRE(hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE(
    "Serialised, async memcpy, null stream.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    const auto test{[&]() {
        vector<float> A(N, 1000.0f);
        vector<float> B(N, 2000.0f);
        vector<float> C(N, -1.f);

        float* A_d{};
        float* B_d{};
        float* C_d{};

        REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipMemcpyAsync(
            A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d, data(B), sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            C_d, data(C), sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        A.assign(size(A), 1.0f);
        B.assign(size(B), 2.0f);
        C.assign(size(C), -1.f);

        const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
        for (auto i = 0u; i != p_iters; ++i) {
            REQUIRE(hipMemcpyAsync(
                A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                B_d, data(B), sizeof(float) * size(B)) == hipSuccess);

            hipLaunchKernelGGL(
                vectorADDReverse,
                dim3(blocks),
                dim3(threadsPerBlock),
                0,
                nullptr,
                A_d,
                B_d,
                C_d,
                N);

            REQUIRE(hipMemcpyAsync(
                data(C), C_d, sizeof(float) * size(C)) == hipSuccess);

            REQUIRE(hipDeviceSynchronize() == hipSuccess);

            for (auto j = 0u; j != size(C); ++j) {
                REQUIRE(Approx{C[j]} == A[j] + B[j]);
            }
        }
    }};

    thread{test}.join();
    thread{test}.join();
}

TEST_CASE(
    "Serialised, async memcpy, different streams.",
    "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    hipStream_t stream0{};
    hipStream_t stream1{};

    REQUIRE(hipStreamCreate(&stream0) == hipSuccess);
    REQUIRE(hipStreamCreate(&stream1) == hipSuccess);

    const auto test{[&](auto&& s) {
        vector<float> A(N, 1000.0f);
        vector<float> B(N, 2000.0f);
        vector<float> C(N, -1.f);

        float* A_d{};
        float* B_d{};
        float* C_d{};

        REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipMemcpyAsync(
            A_d,
            data(A),
            sizeof(float) * size(A),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d,
            data(B),
            sizeof(float) * size(B),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            C_d,
            data(C),
            sizeof(float) * size(C),
            hipMemcpyHostToDevice,
            s) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        A.assign(size(A), 1.0f);
        B.assign(size(B), 2.0f);
        C.assign(size(C), -1.f);

        const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
        for (auto i = 0u; i != p_iters; ++i) {
            REQUIRE(hipMemcpyAsync(
                A_d,
                data(A),
                sizeof(float) * size(A),
                hipMemcpyHostToDevice,
                s) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                B_d,
                data(B),
                sizeof(float) * size(B),
                hipMemcpyHostToDevice,
                s) == hipSuccess);

            hipLaunchKernelGGL(
                vectorADDReverse,
                dim3(blocks),
                dim3(threadsPerBlock),
                0,
                s,
                A_d,
                B_d,
                C_d,
                N);

            REQUIRE(hipMemcpyAsync(
                data(C),
                C_d,
                sizeof(float) * size(C),
                hipMemcpyDeviceToHost,
                s) == hipSuccess);

            REQUIRE(hipDeviceSynchronize() == hipSuccess);

            for (auto j = 0u; j != size(C); ++j) {
                REQUIRE(Approx{C[j]} == A[j] + B[j]);
            }
        }
    }};

    thread{test, stream0}.join();
    thread{test, stream1}.join();

    REQUIRE(hipStreamDestroy(stream0) == hipSuccess);
    REQUIRE(hipStreamDestroy(stream1) == hipSuccess);
}

TEST_CASE("Parallel, async memcpy, null stream.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    const auto test{[&]() {
        vector<float> A(N, 1000.0f);
        vector<float> B(N, 2000.0f);
        vector<float> C(N, -1.f);

        float* A_d{};
        float* B_d{};
        float* C_d{};

        REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipMemcpyAsync(
            A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d, data(B), sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            C_d, data(C), sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        A.assign(size(A), 1.0f);
        B.assign(size(B), 2.0f);
        C.assign(size(C), -1.f);

        const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
        for (auto i = 0u; i != p_iters; ++i) {
            REQUIRE(hipMemcpyAsync(
                A_d, data(A), sizeof(float) * size(A)) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                B_d, data(B), sizeof(float) * size(B)) == hipSuccess);

            hipLaunchKernelGGL(
                vectorADDReverse,
                dim3(blocks),
                dim3(threadsPerBlock),
                0,
                nullptr,
                A_d,
                B_d,
                C_d,
                N);

            REQUIRE(hipMemcpyAsync(
                data(C), C_d, sizeof(float) * size(C)) == hipSuccess);

            REQUIRE(hipDeviceSynchronize() == hipSuccess);

            for (auto j = 0u; j != size(C); ++j) {
                REQUIRE(Approx{C[j]} == A[j] + B[j]);
            }
        }
    }};

    thread t0{test};
    thread t1{test};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());
}

TEST_CASE("Parallel, async memcpy, same stream.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    hipStream_t s{};

    REQUIRE(hipStreamCreate(&s) == hipSuccess);

    const auto test{[=]() {
        vector<float> A(N, 1000.0f);
        vector<float> B(N, 2000.0f);
        vector<float> C(N, -1.f);

        float* A_d{};
        float* B_d{};
        float* C_d{};

        REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipMemcpyAsync(
            A_d,
            data(A),
            sizeof(float) * size(A),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d,
            data(B),
            sizeof(float) * size(B),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            C_d,
            data(C),
            sizeof(float) * size(C),
            hipMemcpyHostToDevice,
            s) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        A.assign(size(A), 1.0f);
        B.assign(size(B), 2.0f);
        C.assign(size(C), -1.f);

        const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
        for (auto i = 0u; i != p_iters; ++i) {
            REQUIRE(hipMemcpyAsync(
                A_d,
                data(A),
                sizeof(float) * size(A),
                hipMemcpyHostToDevice,
                s) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                B_d,
                data(B),
                sizeof(float) * size(B),
                hipMemcpyHostToDevice,
                s) == hipSuccess);

            hipLaunchKernelGGL(
                vectorADDReverse,
                dim3(blocks),
                dim3(threadsPerBlock),
                0,
                s,
                A_d,
                B_d,
                C_d,
                N);

            REQUIRE(hipMemcpyAsync(
                data(C),
                C_d,
                sizeof(float) * size(C),
                hipMemcpyDeviceToHost,
                s) == hipSuccess);

            REQUIRE(hipDeviceSynchronize() == hipSuccess);

            for (auto j = 0u; j != size(C); ++j) {
                REQUIRE(Approx{C[j]} == A[j] + B[j]);
            }
        }
    }};

    thread t0{test};
    thread t1{test};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());

    REQUIRE(hipStreamDestroy(s) == hipSuccess);
}

TEST_CASE(
    "Parallel, async memcpy, different streams.", "[host][multithread][memcpy]")
{
    REQUIRE(hipSetDevice(0) == hipSuccess);
    REQUIRE(hipDeviceReset() == hipSuccess);

    hipStream_t stream0{};
    hipStream_t stream1{};

    REQUIRE(hipStreamCreate(&stream0) == hipSuccess);
    REQUIRE(hipStreamCreate(&stream1) == hipSuccess);

    const auto test{[&](auto&& s) {
        vector<float> A(N, 1000.0f);
        vector<float> B(N, 2000.0f);
        vector<float> C(N, -1.f);

        float* A_d{};
        float* B_d{};
        float* C_d{};

        REQUIRE(hipMalloc(&A_d, sizeof(float) * size(A)) == hipSuccess);
        REQUIRE(hipMalloc(&B_d, sizeof(float) * size(B)) == hipSuccess);
        REQUIRE(hipMalloc(&C_d, sizeof(float) * size(C)) == hipSuccess);

        REQUIRE(hipMemcpyAsync(
            A_d,
            data(A),
            sizeof(float) * size(A),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            B_d,
            data(B),
            sizeof(float) * size(B),
            hipMemcpyHostToDevice,
            s) == hipSuccess);
        REQUIRE(hipMemcpyAsync(
            C_d,
            data(C),
            sizeof(float) * size(C),
            hipMemcpyHostToDevice,
            s) == hipSuccess);

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        A.assign(size(A), 1.0f);
        B.assign(size(B), 2.0f);
        C.assign(size(C), -1.f);

        const auto blocks{set_num_blocks(blocksPerCU, threadsPerBlock, N)};
        for (auto i = 0u; i != p_iters; ++i) {
            REQUIRE(hipMemcpyAsync(
                A_d,
                data(A),
                sizeof(float) * size(A),
                hipMemcpyHostToDevice,
                s) == hipSuccess);
            REQUIRE(hipMemcpyAsync(
                B_d,
                data(B),
                sizeof(float) * size(B),
                hipMemcpyHostToDevice,
                s) == hipSuccess);

            hipLaunchKernelGGL(
                vectorADDReverse,
                dim3(blocks),
                dim3(threadsPerBlock),
                0,
                s,
                A_d,
                B_d,
                C_d,
                N);

            REQUIRE(hipMemcpyAsync(
                data(C),
                C_d,
                sizeof(float) * size(C),
                hipMemcpyDeviceToHost,
                s) == hipSuccess);

            REQUIRE(hipDeviceSynchronize() == hipSuccess);

            for (auto j = 0u; j != size(C); ++j) {
                REQUIRE(Approx{C[j]} == A[j] + B[j]);
            }
        }
    }};

    thread t0{test, stream0};
    thread t1{test, stream1};

    REQUIRE_NOTHROW(t0.join());
    REQUIRE_NOTHROW(t1.join());

    REQUIRE(hipStreamDestroy(stream0) == hipSuccess);
    REQUIRE(hipStreamDestroy(stream1) == hipSuccess);
}