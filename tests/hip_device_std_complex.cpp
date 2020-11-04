/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <utility>
#include <vector>

constexpr auto cnt{64u};

template<typename T, typename F>
__global__
void kernel(std::complex<T>* A, std::complex<T>* B, std::complex<T>* C, F fn)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    C[tx] = fn(A[tx], B[tx]);
}

using namespace std;

struct Cabs {
    template<typename T>
    decltype(auto) operator()(const T& x, const T&) { return abs(x); }
};
struct Carg {
    template<typename T>
    decltype(auto) operator()(const T& x, const T&) { return arg(x); }
};
struct Ccos {
    template<typename T>
    decltype(auto) operator()(const T& x, const T&) { return cos(x); }
};
struct Csin {
    template<typename T>
    decltype(auto) operator()(const T& x, const T&) { return sin(x); }
};

TEMPLATE_TEST_CASE(
    "std::complex",
    "[device][std::complex]",
    (pair<float, plus<>>),
    (pair<float, minus<>>),
    (pair<float, multiplies<>>),
    (pair<float, divides<>>),
    (pair<float, Cabs>),
    (pair<float, Carg>),
    (pair<float, Ccos>),
    (pair<float, Csin>),
    (pair<double, plus<>>),
    (pair<double, minus<>>),
    (pair<double, multiplies<>>),
    (pair<double, divides<>>),
    (pair<double, Cabs>),
    (pair<double, Carg>),
    (pair<double, Ccos>),
    (pair<double, Csin>))
{
    using T = std::complex<typename TestType::first_type>;
    using U = typename T::value_type;
    using F = typename TestType::second_type;

    vector<T> A(cnt);
    vector<T> B(cnt);
    vector<T> C(cnt);
    vector<T> D(cnt);

    T* Ad;
    T* Bd;
    T* Cd;
    REQUIRE(hipMalloc((void**)&Ad, sizeof(T) * cnt) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, sizeof(T) * cnt) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Cd, sizeof(T) * cnt) == hipSuccess);

    for (auto i = 0u; i != cnt; ++i) {
        A[i] = T{(i + 1) * U{1.0}, (i + 2) * U{1.0}};
        B[i] = A[i];
        C[i] = A[i];
    }

    REQUIRE(hipMemcpy(
        Ad, data(A), sizeof(T) * cnt, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hipMemcpy(
        Bd, data(B), sizeof(T) * cnt, hipMemcpyHostToDevice) == hipSuccess);

    hipLaunchKernelGGL(kernel, dim3(1), dim3(cnt), 0, nullptr, Ad, Bd, Cd, F{});

    REQUIRE(hipMemcpy(
        data(C), Cd, sizeof(T) * cnt, hipMemcpyDeviceToHost) == hipSuccess);

    for (auto i = 0u; i != cnt; ++i) {
        const auto expected{F{}(A[i], B[i])};

        REQUIRE(Approx{real(C[i])} == real(expected));
        REQUIRE(Approx{imag(C[i])} == imag(expected));
    }

    REQUIRE(hipFree(Ad) == hipSuccess);
    REQUIRE(hipFree(Bd) == hipSuccess);
    REQUIRE(hipFree(Cd) == hipSuccess);
}