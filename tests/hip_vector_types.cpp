/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_vector_types.h>

#include <catch2/catch.hpp>

#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <type_traits>

using namespace std;

TEST_CASE("HIP Vector Types size", "[host][vector_types]")
{
    REQUIRE(sizeof(float1) == 4);
    REQUIRE(sizeof(float2) >= 8);
    REQUIRE(sizeof(float3) == 12);
    REQUIRE(sizeof(float4) >= 16);
}

TEMPLATE_TEST_CASE(
    "HIP Vector Types",
    "[host][vector_types]",
    char1, char2, char3, char4,
    uchar1, uchar2, uchar3, uchar4,
    short1, short2, short3, short4,
    ushort1, ushort2, ushort3, ushort4,
    int1, int2, int3, int4,
    uint1, uint2, uint3, uint4,
    long1, long2, long3, long4,
    ulong1, ulong2, ulong3, ulong4,
    longlong1, longlong2, longlong3, longlong4,
    ulonglong1, ulonglong2, ulonglong3, ulonglong4,
    float1, float2, float3, float4,
    double1, double2, double3, double4)
{
    SECTION("Unary Constructors") {
        REQUIRE(is_constructible_v<TestType, unsigned char>);
        REQUIRE(is_constructible_v<TestType, signed char>);
        REQUIRE(is_constructible_v<TestType, unsigned short>);
        REQUIRE(is_constructible_v<TestType, signed short>);
        REQUIRE(is_constructible_v<TestType, unsigned int>);
        REQUIRE(is_constructible_v<TestType, signed int>);
        REQUIRE(is_constructible_v<TestType, unsigned long>);
        REQUIRE(is_constructible_v<TestType, signed long>);
        REQUIRE(is_constructible_v<TestType, unsigned long long>);
        REQUIRE(is_constructible_v<TestType, signed long long>);
        REQUIRE(is_constructible_v<TestType, float>);
        REQUIRE(is_constructible_v<TestType, double>);
    }

    SECTION("Common Operators") {
        SECTION("Binary") {
            TestType f1{1};
            TestType f2{1};
            TestType f3 = f1 + f2;

            REQUIRE((f3 == TestType{2}));
            REQUIRE((f3 != f2));
            REQUIRE((f2 == f3 - f1));
            REQUIRE((f3 == f3 * f1));
            REQUIRE((f3 == f3 / f1));
        }

        SECTION("Unary") {
            TestType f1{2};
            TestType f2{1};

            REQUIRE(((f1 += f2) == TestType{3}));
            REQUIRE(((f1 -= f2) == TestType{2}));
            REQUIRE(((f1 *= f2) == TestType{2}));
            REQUIRE(((f1 /= f2) == TestType{2}));
        }

        SECTION("Nullary") {
            TestType f1{2};

            REQUIRE((f1++ == TestType{2} && f1 == TestType{3}));
            REQUIRE((f1-- == TestType{3} && f1 == TestType{2}));
            REQUIRE((++f1 == TestType{3} && f1 == TestType{3}));
            REQUIRE((--f1 == TestType{2} && f1 == TestType{2}));

            using T = typename TestType::value_type;

            const T& x = f1.x;
            T& y = f1.x;
            const volatile T& z = f1.x;
            volatile T& w = f1.x;

            REQUIRE(x == T{2});
            REQUIRE(y == T{2});
            REQUIRE(z == T{2});
            REQUIRE(w == T{2});
        }

        SECTION("Get && Put for components") {
            TestType f1{1};
            TestType f2;

            stringstream str;
            str << f1.x;
            str >> f2.x;

            REQUIRE(f1.x == f2.x);
        }
    }

    if constexpr (is_integral_v<typename TestType::value_type>) {
        SECTION("Integer-only Binary Operators") {
            TestType f1{2};
            TestType f2{1};

            REQUIRE(((f1 % f2) == TestType{0}));
            REQUIRE(((f1 & f2) == TestType{0}));
            REQUIRE(((f1 ^ f2) == TestType{3}));
            REQUIRE(((f1 << TestType{2}) == TestType{8}));
            REQUIRE(((f1 >> f2) == TestType{1}));
        }

        SECTION("Integer-only Unary Operators") {
            TestType f1{4};

            REQUIRE(((f1 %= TestType{3}) == TestType{1}));
            REQUIRE(((f1 &= TestType{2}) == TestType{0}));
            REQUIRE(((f1 ^= TestType{1}) == TestType{1}));
            REQUIRE(((f1 <<= TestType{2}) == TestType{4}));
            REQUIRE(((f1 >>= TestType{2}) == TestType{1}));
        }

        SECTION("Integer-only Nullary Operators") {
            TestType f1{1};

            REQUIRE((~f1 == TestType{~f1.x}));
        }
    }
}