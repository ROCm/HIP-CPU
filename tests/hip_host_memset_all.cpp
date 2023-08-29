/* -----------------------------------------------------------------------------
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <cstdint>
#include <cstdlib>
#include <vector>

#define MAX_OFFSET 3
// To test memset on unaligned pointer
#define loop(offset, offsetMax) \
    for (int offset = offsetMax; offset >= 0; offset --)

enum MemsetType {
    hipMemsetTypeDefault,
    hipMemsetTypeD8,
    hipMemsetTypeD16,
    hipMemsetTypeD32
};

using namespace std;

TEST_CASE("memset small size", "[host][memset][small]")
{
    constexpr auto memsetval{0x42};

    char* A_d;
    char* A_h;
    bool testResult = true;
    for (size_t iSize = 1; iSize < 4; iSize++) {
        size_t Nbytes = iSize * sizeof(char);

        REQUIRE(hipMalloc(&A_d, Nbytes) == hipSuccess);

        A_h = reinterpret_cast<char*>(malloc(Nbytes));

        INFO(
            "testhipMemsetSmallSize N = " << iSize
                << " memsetval = " << memsetval);

        REQUIRE(hipMemset(A_d, memsetval, Nbytes) == hipSuccess);
        REQUIRE(hipMemcpy(
            A_h, A_d, Nbytes, hipMemcpyDeviceToHost) == hipSuccess);

        for (int i = 0; i < iSize; i++) REQUIRE(A_h[i] == memsetval);

        REQUIRE(hipFree(A_d) == hipSuccess);

        free(A_h);
    }
}

void testhipMemset(size_t N, MemsetType type)
{
    const auto memsetval{(N == 10) ? 0x42 : ((N == 10013) ? 0x5a : 0xa6)};
    const auto memsetD8val{(N == 10) ? 0x1 : ((N == 10013) ? 0xDE : 0xCA)};
    const auto memsetD16val{
        (N == 10) ? 0x10 : ((N == 10013) ? 0xDEAD : 0xCAFE)};
    const auto memsetD32val{
        (N == 10) ? 0x101 : ((N == 10013) ? 0xDEADBEEF : 0xCAFEBABE)};
    const auto elt_sz{
        (type == hipMemsetTypeD32) ? sizeof(uint32_t) :
            ((type == hipMemsetTypeD16) ? sizeof(uint16_t) : sizeof(uint8_t))};
    const auto Nbytes = N * elt_sz;

    auto A_h{malloc(Nbytes)};

    void* A_d{};
    REQUIRE(hipMalloc(&A_d, Nbytes) == hipSuccess);

    loop(offset, MAX_OFFSET) {
        switch (type) {
        case hipMemsetTypeDefault:
            REQUIRE(hipMemset(
                static_cast<uint8_t*>(A_d) + offset,
                memsetval,
                N - offset) == hipSuccess);
            break;
        case hipMemsetTypeD8:
            REQUIRE(hipMemsetD8(
                static_cast<uint8_t*>(A_d) + offset,
                memsetD8val,
                N - offset) == hipSuccess);
            break;
        case hipMemsetTypeD16:
            REQUIRE(hipMemsetD16(
                static_cast<uint16_t*>(A_d) + offset,
                memsetD16val,
                N - offset) == hipSuccess);
            break;
        case hipMemsetTypeD32:
            REQUIRE(hipMemsetD32(
                static_cast<uint32_t*>(A_d) + offset,
                memsetD32val,
                N - offset) == hipSuccess);
            break;
        default:
            FAIL();
        }

        REQUIRE(
            hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost) == hipSuccess);

        for (int i = offset; i < N; i++) {
            switch (type) {
            case hipMemsetTypeDefault:
                REQUIRE(static_cast<uint8_t*>(A_h)[i] == memsetval);
                break;
            case hipMemsetTypeD8:
                REQUIRE(static_cast<uint8_t*>(A_h)[i] == memsetD8val);
                break;
            case hipMemsetTypeD16:
                REQUIRE(static_cast<uint16_t*>(A_h)[i] == memsetD16val);
                break;
            case hipMemsetTypeD32:
                REQUIRE(static_cast<uint32_t*>(A_h)[i] == memsetD32val);
                break;
            default:
                FAIL();
            }
        }
    }

    REQUIRE(hipFree(A_d) == hipSuccess);

    free(A_h);
}

TEST_CASE("memset test types", "[host][memset][types]")
{
    const auto N{GENERATE(10, 10013, 256 * 1024 * 1024)};
    const auto T{GENERATE(
        hipMemsetTypeDefault,
        hipMemsetTypeD8,
        hipMemsetTypeD16,
        hipMemsetTypeD32)};

    testhipMemset(N, T);
}

TEST_CASE("memsetAsync 2 ops at the same time", "[host][memset][async]")
{
  vector<float> v(2048);

  float* p2;

  REQUIRE(hipMalloc(&p2, 4096 + 4096 * 2) == hipSuccess);

  float* p3 = p2 + 2048;

  hipStream_t s{};
  REQUIRE(hipStreamCreate(&s) == hipSuccess);

  REQUIRE(hipMemsetAsync(p2, 0, 32 * 32 * 4, s) == hipSuccess);
  REQUIRE(hipMemsetD32Async(p3, 0x3fe00000, 32 * 32, s) == hipSuccess);
  REQUIRE(hipStreamSynchronize(s) == hipSuccess);

  for (int i = 0; i < 256; ++i) {
    REQUIRE(hipMemsetAsync(p2, 0, 32 * 32 * 4, s) == hipSuccess);
    REQUIRE(hipMemsetD32Async(p3, 0x3fe00000, 32 * 32, s) == hipSuccess);
  }
  REQUIRE(hipStreamSynchronize(s) == hipSuccess);
  REQUIRE(hipDeviceSynchronize() == hipSuccess);

  REQUIRE(hipMemcpy(&v[0], p2, 1024, hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(&v[1024], p3, 1024, hipMemcpyDeviceToHost) == hipSuccess);

  REQUIRE(v[0] == 0);
  REQUIRE(v[1024] == 1.75f);
}

void testhipMemsetAsync(size_t N, MemsetType type)
{
    const auto memsetval{(N == 10) ? 0x42 : ((N == 10013) ? 0x5a : 0xa6)};
    const auto memsetD8val{(N == 10) ? 0x1 : ((N == 10013) ? 0xDE : 0xCA)};
    const auto memsetD16val{
        (N == 10) ? 0x10 : ((N == 10013) ? 0xDEAD : 0xCAFE)};
    const auto memsetD32val{
        (N == 10) ? 0x101 : ((N == 10013) ? 0xDEADBEEF : 0xCAFEBABE)};
    const auto elt_sz{
        (type == hipMemsetTypeD32) ? sizeof(uint32_t) :
            ((type == hipMemsetTypeD16) ? sizeof(uint16_t) : sizeof(uint8_t))};
    const auto Nbytes = N * elt_sz;

    auto A_h{malloc(Nbytes)};

    void* A_d{};
    REQUIRE(hipMalloc(&A_d, Nbytes) == hipSuccess);

    hipStream_t stream{};
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    loop(offset, MAX_OFFSET) {
        switch (type) {
        case hipMemsetTypeDefault:
            REQUIRE(hipMemsetAsync(
                static_cast<uint8_t*>(A_d) + offset,
                memsetval,
                N - offset, stream) == hipSuccess);
            break;
        case hipMemsetTypeD8:
            REQUIRE(hipMemsetD8Async(
                static_cast<uint8_t*>(A_d) + offset,
                memsetD8val,
                N - offset,
                stream) == hipSuccess);
            break;
        case hipMemsetTypeD16:
            REQUIRE(hipMemsetD16Async(
                static_cast<uint16_t*>(A_d) + offset,
                memsetD16val,
                N - offset,
                stream) == hipSuccess);
            break;
        case hipMemsetTypeD32:
            REQUIRE(hipMemsetD32Async(
                static_cast<uint32_t*>(A_d) + offset,
                memsetD32val,
                N - offset,
                stream) == hipSuccess);
            break;
        default:
            FAIL();
        }

        REQUIRE(hipStreamSynchronize(stream) == hipSuccess);
        REQUIRE(
            hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost) == hipSuccess);

        for (int i = offset; i < N; i++) {
            switch (type) {
            case hipMemsetTypeDefault:
                REQUIRE(static_cast<uint8_t*>(A_h)[i] == memsetval);
                break;
            case hipMemsetTypeD8:
                REQUIRE(static_cast<uint8_t*>(A_h)[i] == memsetD8val);
                break;
            case hipMemsetTypeD16:
                REQUIRE(static_cast<uint16_t*>(A_h)[i] == memsetD16val);
                break;
            case hipMemsetTypeD32:
                REQUIRE(static_cast<uint32_t*>(A_h)[i] == memsetD32val);
                break;
            default:
                FAIL();
            }
        }
    }

    REQUIRE(hipFree(A_d) == hipSuccess);
    REQUIRE(hipStreamDestroy(stream) == hipSuccess);

    free(A_h);
}

TEST_CASE("memset_async test types", "[host][memset][types][async]")
{
    const auto N{GENERATE(10, 10013, 256 * 1024 * 1024)};
    const auto T{GENERATE(
        hipMemsetTypeDefault,
        hipMemsetTypeD8,
        hipMemsetTypeD16,
        hipMemsetTypeD32)};

    testhipMemsetAsync(N, T);
}