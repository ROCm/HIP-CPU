/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

constexpr auto N{512};
const auto pi{std::atan(1) * 4};

__global__
void test_sincosf(float* a, float* b, float* c)
{
    int tid = threadIdx.x;
    sincosf(a[tid], b + tid, c + tid);
}

__global__
void test_sincospif(float* a, float* b, float* c)
{
    int tid = threadIdx.x;
    sincospif(a[tid], b + tid, c + tid);
}

__global__
void test_fdividef(float* a, float* b, float* c)
{
    int tid = threadIdx.x;
    c[tid] = fdividef(a[tid], b[tid]);
}

__global__
void test_llrintf(float* a, long long int* b)
{
    int tid = threadIdx.x;
    b[tid] = llrintf(a[tid]);
}

__global__
void test_lrintf(float* a, long int* b)
{
    int tid = threadIdx.x;
    b[tid] = lrintf(a[tid]);
}

__global__
void test_rintf(float* a, float* b)
{
    int tid = threadIdx.x;
    b[tid] = rintf(a[tid]);
}

__global__
void test_llroundf(float* a, long long int* b)
{
    int tid = threadIdx.x;
    b[tid] = llroundf(a[tid]);
}

__global__
void test_lroundf(float* a, long int* b)
{
    int tid = threadIdx.x;
    b[tid] = lroundf(a[tid]);
}

__global__
void test_rhypotf(float* a, float* b, float* c)
{
    int tid = threadIdx.x;
    c[tid] = rhypotf(a[tid], b[tid]);
}

__global__
void test_norm3df(float* a, float* b, float* c, float* d)
{
    int tid = threadIdx.x;
    d[tid] = norm3df(a[tid], b[tid], c[tid]);
}

__global__
void test_norm4df(float* a, float* b, float* c, float* d, float* e)
{
    int tid = threadIdx.x;
    e[tid] = norm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__
void test_normf(float* a, float* b)
{
    int tid = threadIdx.x;
    b[tid] = normf(N, a);
}

__global__
void test_rnorm3df(float* a, float* b, float* c, float* d)
{
    int tid = threadIdx.x;
    d[tid] = rnorm3df(a[tid], b[tid], c[tid]);
}

__global__
void test_rnorm4df(float* a, float* b, float* c, float* d, float* e)
{
    int tid = threadIdx.x;
    e[tid] = rnorm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__
void test_rnormf(float* a, float* b)
{
    int tid = threadIdx.x;
    b[tid] = rnormf(N, a);
}

__global__
void test_erfinvf(float* a, float* b)
{
    int tid = threadIdx.x;
    b[tid] = erff(erfinvf(a[tid]));
}

using namespace std;

TEST_CASE("sincosf()", "[device][math][sincosf]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N);
    vector<float> C(N);

    float* Ad;
    float* Bd;
    float* Cd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_sincosf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        data(C), Cd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(B), cend(B), [](auto&& x) { return Approx{x} == sinf(1); }));
    REQUIRE(all_of(cbegin(C), cend(C), [](auto&& x) { return Approx{x} == cosf(1); }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("sincospif()", "[device][math][sincospif]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N);
    vector<float> C(N);

    float* Ad;
    float* Bd;
    float* Cd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_sincospif, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        data(C), Cd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(B), cend(B), [](auto&& x) { return Approx{x} == sinf(pi); }));
    REQUIRE(all_of(cbegin(C), cend(C), [](auto&& x) { return Approx{x} == cosf(pi); }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("fdividef", "[device][math][fdividef]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N);

    float* Ad;
    float* Bd;
    float* Cd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_fdividef, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(C), Cd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    for (auto i = 0u; i != size(C); ++i) REQUIRE(Approx{C[i]} == A[i] / B[i]);

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("llrintf", "[device][math][llrintf]")
{
    vector<float> A(N, 1.345f);
    vector<long long> B(N);

    float* Ad;
    long long* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(long long) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_llrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long long) * N,
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(
        cbegin(A),
        cend(A),
        cbegin(B),
        [](auto&& x, auto&& y) { return llrintf(x) == y; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("lrintf()", "[device][math][lrintf]")
{
    vector<float> A(N, 1.345f);
    vector<long int> B(N);

    float* Ad;
    long* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(long int) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_lrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(long) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(
        cbegin(A),
        cend(A),
        cbegin(B),
        [](auto&& x, auto&& y) { return lrintf(x) == y; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("rintf", "[device][math][rintf]")
{
    vector<float> A(N, 1.345f);
    vector<float> B(N);

    float* Ad;
    float* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rintf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(
        cbegin(A),
        cend(A),
        cbegin(B),
        [](auto&& x, auto&& y) { return rintf(x) == y; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("llroundf", "[device][math][llroundf]")
{
    vector<float> A(N, 1.345f);
    vector<long long> B(N);

    float* Ad;
    long long* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(long long) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_llroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long long) * N,
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(
        cbegin(A),
        cend(A),
        cbegin(B),
        [](auto&& x, auto&& y) { return llroundf(x) == y; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("lroundf()", "[device][math][lroundf]")
{
    vector<float> A(N, 1.345f);
    vector<long int> B(N);

    float* Ad;
    long* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(long int) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_lroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(long) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(
        cbegin(A),
        cend(A),
        cbegin(B),
        [](auto&& x, auto&& y) { return lroundf(x) == y; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("norm3df()", "[device][math][norm3df]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N, 3.0f);
    vector<float> D(N);

    float* Ad;
    float* Bd;
    float* Cd;
    float* Dd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd, data(C), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_norm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);

    REQUIRE(hipMemcpy(
        data(D), Dd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{sqrtf(1.0f + 4.0f + 9.0f)};
    REQUIRE(
        all_of(cbegin(D), cend(D), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
}

TEST_CASE("norm4df()", "[device][math][norm4df]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N, 3.0f);
    vector<float> D(N, 4.0f);
    vector<float> E(N);

    float val = 0.0f;
    val = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);

    float* Ad;
    float* Bd;
    float* Cd;
    float* Dd;
    float* Ed;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Ed, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd, data(C), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Dd, data(D), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        test_norm4df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);

    REQUIRE(hipMemcpy(
        data(E), Ed, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{sqrtf(1.0f + 4.0f + 9.0f + 16.0f)};
    REQUIRE(
        all_of(cbegin(E), cend(E), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
    REQUIRE(hipFree(Ed) == hipSuccess);
}

TEST_CASE("normf", "[device][math][normf]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 0.0f);

    float* Ad;
    float* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_normf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{sqrtf(1.0f * N)};
    REQUIRE(
        all_of(cbegin(B), cend(B), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("rhypotf", "[device][math][rhypotf]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N);

    float* Ad;
    float* Bd;
    float* Cd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rhypotf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(C), Cd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{1 / sqrtf(1.0f + 4.0f)};
    REQUIRE(
        all_of(cbegin(C), cend(C), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("rnorm3df", "[device][math][rnorm3df]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N, 3.0f);
    vector<float> D(N);

    float* Ad;
    float* Bd;
    float* Cd;
    float* Dd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd, data(C), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rnorm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);

    REQUIRE(hipMemcpy(
        data(D), Dd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{1 / sqrtf(1.0f + 4.0f + 9.0f)};
    REQUIRE(
        all_of(cbegin(D), cend(D), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
}

TEST_CASE("rnorm4df()", "[device][math][rnorm4df]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 2.0f);
    vector<float> C(N, 3.0f);
    vector<float> D(N, 4.0f);
    vector<float> E(N);

    float val = 0.0f;
    val = 1 / sqrtf(1.0f + 4.0f + 9.0f + 16.0f);

    float* Ad;
    float* Bd;
    float* Cd;
    float* Dd;
    float* Ed;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Dd, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Ed, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd, data(C), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Dd, data(D), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        test_rnorm4df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd, Ed);

    REQUIRE(hipMemcpy(
        data(E), Ed, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{1 / sqrtf(1.0f + 4.0f + 9.0f + 16.0f)};
    REQUIRE(
        all_of(cbegin(E), cend(E), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
    REQUIRE(hipFree(Ed) == hipSuccess);
}

TEST_CASE("rnormf", "[device][math][rnormf]")
{
    vector<float> A(N, 1.0f);
    vector<float> B(N, 0.0f);

    float* Ad;
    float* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rnormf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    const Approx expected{1 / sqrtf(N)};
    REQUIRE(
        all_of(cbegin(B), cend(B), [=](auto&& x) { return x == expected; }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("erfinvf()", "[device][math][erfinvf]")
{
    vector<float> A(N, -0.6f);
    vector<float> B(N);

    float* Ad;
    float* Bd;

    REQUIRE(hipMalloc((void**)&Ad, sizeof(float) * N) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(float) * N) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(float) * N, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_erfinvf, dim3(1), dim3(N), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B), Bd, sizeof(float) * N, hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return Approx{x}.epsilon(0.0009) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}