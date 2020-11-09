/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cfloat>
#include <chrono>
#include <utility>
#include <vector>

constexpr auto iterations{10};
constexpr auto N{4u * 1024u * 1024u};
constexpr auto threads_per_block{256u};

template<typename T>
__global__
void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM)
{
    const auto offset{(blockIdx.x * blockDim.x + threadIdx.x)};
    const auto stride{blockDim.x * gridDim.x};

    for (auto i = offset; i < NELEM; i += stride) C_d[i] = A_d[i] + B_d[i];
}

TEST_CASE("hipEventRecord()", "[host][hipEvent_t][event_record]")
{
    using namespace std;

    vector<float> A(N);
    for (auto i = 0u; i != size(A); ++i) A[i] = 3.146f * i;
    vector<float> B(N);
    for (auto i = 0u; i != size(B); ++i) B[i] = 1.618f * i;
    vector<float> C(N);
    for (auto i = 0u; i != size(C); ++i) C[i] = static_cast<float>(i);

    float* A_d;
    float* B_d;
    float* C_d;

    REQUIRE(hipMalloc((void**)&A_d, size(A) * sizeof(float)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&B_d, size(B) * sizeof(float)) == hipSuccess);
    REQUIRE(hipMalloc((void**)&C_d, size(C) * sizeof(float)) == hipSuccess);

    hipEvent_t start;
    hipEvent_t stop;

    REQUIRE(hipEventCreate(&start) == hipSuccess);
    REQUIRE(hipEventCreate(&stop) == hipSuccess);

    REQUIRE(hipMemcpy(
        A_d,
        data(A),
        size(A) * sizeof(float),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        B_d,
        data(B),
        size(B) * sizeof(float),
        hipMemcpyHostToDevice) == hipSuccess);

    const dim3 blocks{(N + threads_per_block - 1) / threads_per_block};
    for (int i = 0; i < iterations; i++) {
        const auto hostStart{chrono::high_resolution_clock::now()};

        REQUIRE(hipEventRecord(start, nullptr) == hipSuccess);

        hipLaunchKernelGGL(
            vectorADD, blocks, dim3(threads_per_block), 0, 0, A_d, B_d, C_d, N);

        REQUIRE(hipEventRecord(stop, nullptr) == hipSuccess);
        REQUIRE(hipEventSynchronize(stop) == hipSuccess);

        const auto hostStop{chrono::high_resolution_clock::now()};

        auto eventMs{FLT_MIN};
        REQUIRE(hipEventElapsedTime(&eventMs, start, stop) == hipSuccess);

        using MS =
            std::chrono::duration<float, std::chrono::milliseconds::period>;
        const auto hostMs{
            chrono::duration_cast<MS>(hostStop - hostStart).count()};

        REQUIRE(eventMs > 0.f);
        REQUIRE(eventMs <= Approx{hostMs});
    }


    REQUIRE(hipMemcpy(
        data(C),
        C_d,
        size(C) * sizeof(float),
        hipMemcpyDeviceToHost) == hipSuccess);

    for (decltype(size(C)) i = 0u; i != size(C); ++i) {
        REQUIRE(Approx{C[i]} == A[i] + B[i]);
    }

    REQUIRE(hipFree(A_d) == hipSuccess);
    REQUIRE(hipFree(B_d) == hipSuccess);
    REQUIRE(hipFree(C_d) == hipSuccess);
}