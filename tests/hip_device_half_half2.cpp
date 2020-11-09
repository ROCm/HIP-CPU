/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <type_traits>

using namespace std;

__global__
void __halfTest(bool* result, __half a)
{
    // Nullary
    result[0] = __heq(a, +a) && result[0];
    result[0] = __heq(__hneg(a), -a) && result[0];

    // Unary arithmetic
    result[0] = __heq(a += 0, a) && result[0];
    result[0] = __heq(a -= 0, a) && result[0];
    result[0] = __heq(a *= 1, a) && result[0];
    result[0] = __heq(a /= 1, a) && result[0];

    // Binary arithmetic
    result[0] = __heq((a + a), __hadd(a, a)) && result[0];
    result[0] = __heq((a - a), __hsub(a, a)) && result[0];
    result[0] = __heq((a * a), __hmul(a, a)) && result[0];
    result[0] = __heq((a / a), __hdiv(a, a)) && result[0];

    // Relations
    result[0] = (a == a) && result[0];
    result[0] = !(a != a) && result[0];
    result[0] = (a <= a) && result[0];
    result[0] = (a >= a) && result[0];
    result[0] = !(a < a) && result[0];
    result[0] = !(a > a) && result[0];
}

__device__
bool to_bool(const __half2& x) noexcept
{
    return x.x != __half{0} && x.y != __half{0};
}

__global__
void __half2Test(bool* result, __half2 a) {
    // Nullary
    result[0] = to_bool(__heq2(a, +a)) && result[0];
    result[0] = to_bool(__heq2(__hneg2(a), -a)) && result[0];

    // Unary arithmetic
    result[0] = to_bool(__heq2(a += 0, a)) && result[0];
    result[0] = to_bool(__heq2(a -= 0, a)) && result[0];
    result[0] = to_bool(__heq2(a *= 1, a)) && result[0];
    result[0] = to_bool(__heq2(a /= 1, a)) && result[0];

    // Binary arithmetic
    result[0] = to_bool(__heq2((a + a), __hadd2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a - a), __hsub2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a * a), __hmul2(a, a))) && result[0];
    result[0] = to_bool(__heq2((a / a), __h2div(a, a))) && result[0];

    // Relations
    result[0] = (a == a) && result[0];
    result[0] = !(a != a) && result[0];
    result[0] = (a <= a) && result[0];
    result[0] = (a >= a) && result[0];
    result[0] = !(a < a) && result[0];
    result[0] = !(a > a) && result[0];
}

TEST_CASE("__half and __half2 support", "[device][half]")
{
    bool* result{};

    REQUIRE(hipHostMalloc((void**)&result, 1) == hipSuccess);

    result[0] = true;

    SECTION("__half") {
        // Construction
        REQUIRE(is_default_constructible_v<__half>);
        REQUIRE(is_copy_constructible_v<__half>);
        REQUIRE(is_move_constructible_v<__half>);
        REQUIRE(is_constructible_v<__half, float>);
        REQUIRE(is_constructible_v<__half, double>);
        REQUIRE(is_constructible_v<__half, unsigned short>);
        REQUIRE(is_constructible_v<__half, short>);
        REQUIRE(is_constructible_v<__half, unsigned int>);
        REQUIRE(is_constructible_v<__half, int>);
        REQUIRE(is_constructible_v<__half, unsigned long>);
        REQUIRE(is_constructible_v<__half, long>);
        REQUIRE(is_constructible_v<__half, long long>);
        REQUIRE(is_constructible_v<__half, unsigned long long>);

        // Assignment
        REQUIRE(is_copy_assignable_v<__half>);
        REQUIRE(is_move_assignable_v<__half>);
        REQUIRE(is_assignable_v<__half, float>);
        REQUIRE(is_assignable_v<__half, double>);
        REQUIRE(is_assignable_v<__half, unsigned short>);
        REQUIRE(is_assignable_v<__half, short>);
        REQUIRE(is_assignable_v<__half, unsigned int>);
        REQUIRE(is_assignable_v<__half, int>);
        REQUIRE(is_assignable_v<__half, unsigned long>);
        REQUIRE(is_assignable_v<__half, long>);
        REQUIRE(is_assignable_v<__half, long long>);
        REQUIRE(is_assignable_v<__half, unsigned long long>);

        // Conversion
        REQUIRE(is_convertible_v<__half, float>);
        REQUIRE(is_convertible_v<__half, unsigned short>);
        REQUIRE(is_convertible_v<__half, short>);
        REQUIRE(is_convertible_v<__half, unsigned int>);
        REQUIRE(is_convertible_v<__half, int>);
        REQUIRE(is_convertible_v<__half, unsigned long>);
        REQUIRE(is_convertible_v<__half, long>);
        REQUIRE(is_convertible_v<__half, long long>);
        REQUIRE(is_convertible_v<__half, bool>);
        REQUIRE(is_convertible_v<__half, unsigned long long>);

        hipLaunchKernelGGL(
            __halfTest, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, result, __half{1});

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        REQUIRE(result[0]);
    }

    SECTION("__half2") {
        // Construction
        REQUIRE(is_default_constructible_v<__half2>);
        REQUIRE(is_copy_constructible_v<__half2>);
        REQUIRE(is_move_constructible_v<__half2>);
        REQUIRE(is_constructible_v<__half2, __half, __half>);

        // Assignment
        REQUIRE(is_copy_assignable_v<__half2>);
        REQUIRE(is_move_assignable_v<__half2>);

        hipLaunchKernelGGL(
            __half2Test, dim3(1), dim3(1), 0, 0, result, __half2{1, 1});

        REQUIRE(hipDeviceSynchronize() == hipSuccess);

        REQUIRE(result[0]);
    }

    REQUIRE(hipHostFree(result) == hipSuccess);
}