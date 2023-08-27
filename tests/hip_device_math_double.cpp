/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

constexpr auto cnt{512u};

__global__
void test_sincos(double* a, double* b, double* c)
{
    int tid = threadIdx.x;
    sincos(a[tid], b + tid, c + tid);
}

__global__
void test_sincospi(double* a, double* b, double* c)
{
    int tid = threadIdx.x;
    sincospi(a[tid], b + tid, c + tid);
}

__global__
void test_llrint(double* a, long long int* b)
{
    int tid = threadIdx.x;
    b[tid] = llrint(a[tid]);
}

__global__
void test_lrint(double* a, long int* b)
{
    int tid = threadIdx.x;
    b[tid] = lrint(a[tid]);
}

__global__
void test_rint(double* a, double* b)
{
    int tid = threadIdx.x;
    b[tid] = rint(a[tid]);
}

__global__
void test_llround(double* a, long long int* b)
{
    int tid = threadIdx.x;
    b[tid] = llround(a[tid]);
}

__global__
void test_lround(double* a, long int* b)
{
    int tid = threadIdx.x;
    b[tid] = lround(a[tid]);
}

__global__
void test_rhypot(double* a, double* b, double* c)
{
    int tid = threadIdx.x;
    c[tid] = rhypot(a[tid], b[tid]);
}

__global__
void test_norm3d(double* a, double* b, double* c, double* d)
{
    int tid = threadIdx.x;
    d[tid] = norm3d(a[tid], b[tid], c[tid]);
}

__global__
void test_norm4d(double* a, double* b, double* c, double* d, double* e)
{
    int tid = threadIdx.x;
    e[tid] = norm4d(a[tid], b[tid], c[tid], d[tid]);
}

__global__
void test_rnorm3d(double* a, double* b, double* c, double* d)
{
    int tid = threadIdx.x;
    d[tid] = rnorm3d(a[tid], b[tid], c[tid]);
}

__global__
void test_rnorm4d(double* a, double* b, double* c, double* d, double* e)
{
    int tid = threadIdx.x;
    e[tid] = rnorm4d(a[tid], b[tid], c[tid], d[tid]);
}

__global__
void test_rnorm(double* a, double* b)
{
    int tid = threadIdx.x;
    b[tid] = rnorm(cnt, a);
}

__global__
void test_erfinv(double* a, double* b)
{
    int tid = threadIdx.x;
    b[tid] = erf(erfinv(a[tid]));
}

using namespace std;

TEST_CASE("sincos(double)", "[device][math][double][sincos]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt);
    vector<double> C(cnt);

    double* Ad;
    double* Bd;
    double* Cd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_sincos, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(double) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        data(C),
        Cd,
        sizeof(double) * size(C),
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(all_of(cbegin(B), cend(B), [](auto&& x) {
        return Approx{x} == std::sin(1.0);
    }));
    REQUIRE(all_of(cbegin(C), cend(C), [](auto&& x) {
        return Approx{x} == std::cos(1.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("sincospi(double)", "[device][math][double][sincospi]")
{
    static constexpr auto pi{3.14159265358979323846};

    vector<double> A(cnt, 1.0);
    vector<double> B(cnt);
    vector<double> C(cnt);

    double* Ad;
    double* Bd;
    double* Cd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_sincospi, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(double) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(hipMemcpy(
        data(C),
        Cd,
        sizeof(double) * size(C),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(B), cend(B), [](auto&& x) {
        return Approx{x} == std::sin(pi);
    }));
    REQUIRE(all_of(cbegin(C), cend(C), [](auto&& x) {
        return Approx{x} == std::cos(pi);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}


TEST_CASE("llrint(double)", "[device][math][double][llrint]")
{
    vector<double> A(cnt, 1.345);
    vector<long long int> B(cnt);

    double* Ad;
    long long* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(long long) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_llrint, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long long) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return std::llrint(x) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("lrint(double)", "[device][math][double][lrint]")
{
    vector<double> A(cnt, 1.345);
    vector<long int> B(cnt);

    double* Ad;
    long int* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(long int) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_lrint, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long int) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return std::lrint(x) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("rint(double)", "[device][math][double][rint]")
{
    vector<double> A(cnt, 1.345);
    vector<double> B(cnt);

    double* Ad;
    double* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rint, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(double) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return std::rint(x) == Approx{y};
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}


TEST_CASE("llround(double)", "[device][math][double][llround]")
{
    vector<double> A(cnt, 1.345);
    vector<long long int> B(cnt);

    double* Ad;
    long long int* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(long long) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_llround, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long long) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return std::llround(x) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("lround(double)", "[device][math][double][lround]")
{
    vector<double> A(cnt, 1.345);
    vector<long int> B(cnt);

    double* Ad;
    long int* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(long int) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_lround, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(long int) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return std::lround(x) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}


TEST_CASE("norm3d(double)", "[device][math][double][norm3d]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 2.0);
    vector<double> C(cnt, 3.0);
    vector<double> D(cnt);

    double* Ad;
    double* Bd;
    double* Cd;
    double* Dd;

    double val = 0.0;
    val = sqrt(1.0 + 4.0 + 9.0);

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);
    REQUIRE(hipMalloc(&Dd, sizeof(double) * size(D)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        sizeof(double) * size(B),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd,
        data(C),
        sizeof(double) * size(C),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_norm3d, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd, Dd);

    REQUIRE(hipMemcpy(
        data(D),
        Dd,
        sizeof(double) * size(D),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(D), cend(D), [](auto&& x) {
        return Approx{x} == std::sqrt(1.0 + 4.0 + 9.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
}

TEST_CASE("norm4d(double)", "[device][math][double][norm4d]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 2.0);
    vector<double> C(cnt, 3.0);
    vector<double> D(cnt, 4.0);
    vector<double> E(cnt);

    double* Ad;
    double* Bd;
    double* Cd;
    double* Dd;
    double* Ed;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);
    REQUIRE(hipMalloc(&Dd, sizeof(double) * size(D)) == hipSuccess);
    REQUIRE(hipMalloc(&Ed, sizeof(double) * size(E)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        sizeof(double) * size(B),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd,
        data(C),
        sizeof(double) * size(C),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Dd,
        data(D),
        sizeof(double) * size(D),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        test_norm4d, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd, Dd, Ed);

    REQUIRE(hipMemcpy(
        data(E),
        Ed,
        sizeof(double) * size(E),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(E), cend(E), [](auto&& x) {
        return Approx{x} == std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
    REQUIRE(hipFree(Ed) == hipSuccess);
}


TEST_CASE("rhypot(double)", "[device][math][double][rhypot]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 2.0);
    vector<double> C(cnt);

    double* Ad;
    double* Bd;
    double* Cd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        sizeof(double) * size(B),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rhypot, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd);

    REQUIRE(hipMemcpy(
        data(C),
        Cd,
        sizeof(double) * size(C),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(C), cend(C), [](auto&& x) {
        return Approx{x} == 1. / std::sqrt(1.0 + 4.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}

TEST_CASE("rnorm3d(double)", "[device][math][double][rnorm3d]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 2.0);
    vector<double> C(cnt, 3.0);
    vector<double> D(cnt);

    double val = 0.0;
    val = 1 / sqrt(1.0 + 4.0 + 9.0);

    double* Ad;
    double* Bd;
    double* Cd;
    double* Dd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);
    REQUIRE(hipMalloc(&Dd, sizeof(double) * size(D)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        sizeof(double) * size(B),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd,
        data(C),
        sizeof(double) * size(C),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rnorm3d, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd, Dd);

    REQUIRE(hipMemcpy(
        data(D),
        Dd,
        sizeof(double) * size(D),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(D), cend(D), [](auto&& x) {
        return Approx{x} == 1. / std::sqrt(1.0 + 4.0 + 9.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
}

TEST_CASE("rnorm4d(double)", "[device][math][double][rnorm4d]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 2.0);
    vector<double> C(cnt, 3.0);
    vector<double> D(cnt, 4.0);
    vector<double> E(cnt);

    double* Ad;
    double* Bd;
    double* Cd;
    double* Dd;
    double* Ed;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);
    REQUIRE(hipMalloc(&Cd, sizeof(double) * size(C)) == hipSuccess);
    REQUIRE(hipMalloc(&Dd, sizeof(double) * size(D)) == hipSuccess);
    REQUIRE(hipMalloc(&Ed, sizeof(double) * size(E)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd,
        data(B),
        sizeof(double) * size(B),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Cd,
        data(C),
        sizeof(double) * size(C),
        hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Dd,
        data(D),
        sizeof(double) * size(D),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(
        test_rnorm4d, dim3(1), dim3(cnt), 0, 0, Ad, Bd, Cd, Dd, Ed);

    REQUIRE(hipMemcpy(
        data(E),
        Ed,
        sizeof(double) * size(E),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(E), cend(E), [](auto&& x) {
        return Approx{x} == 1. / std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
    REQUIRE(hipFree(Dd) == hipSuccess);
    REQUIRE(hipFree(Ed) == hipSuccess);
}

TEST_CASE("rnorm(double)", "[device][math][double][rnorm]")
{
    vector<double> A(cnt, 1.0);
    vector<double> B(cnt, 0.0);

    double val = 0.0;
    val = 1 / sqrt(val);

    double* Ad;
    double* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_rnorm, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(double) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(all_of(cbegin(B), cend(B), [](auto&& x) {
        return Approx{x} == 1. / sqrt(cnt);
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}

TEST_CASE("erfinv(double)", "[device][math][double][erfinv]")
{
    vector<double> A(cnt, -0.6);
    vector<double> B(cnt, 0.0);

    double* Ad;
    double* Bd;

    REQUIRE(hipMalloc(&Ad, sizeof(double) * size(A)) == hipSuccess);
    REQUIRE(hipMalloc(&Bd, sizeof(double) * size(B)) == hipSuccess);

    REQUIRE(hipMemcpy(
        Ad,
        data(A),
        sizeof(double) * size(A),
        hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(test_erfinv, dim3(1), dim3(cnt), 0, 0, Ad, Bd);

    REQUIRE(hipMemcpy(
        data(B),
        Bd,
        sizeof(double) * size(B),
        hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B), [](auto&& x, auto&& y) {
        return Approx{x}.epsilon(0.0009) == y;
    }));

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
}