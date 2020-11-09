/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include "../external/catch2/catch.hpp"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

using namespace std;

static constexpr auto len{64};
static constexpr auto sz{len << 2};

static constexpr const char fileName[]{"vcpy_kernel.code"};
static constexpr const char kernel_name[]{"hello_world"};

TEST_CASE("load module and launch __global__ function", "[host][module]")
{
    vector<float> A(len); iota(begin(A), end(A), 0.f);
    vector<float> B(len); fill_n(begin(B), size(B), 0.f);

    REQUIRE(hipInit(0) == hipSuccess);

    hipDevice_t device;
    hipCtx_t context;
    REQUIRE(hipDeviceGet(&device, 0) == hipSuccess);
    REQUIRE(hipCtxCreate(&context, 0, device) == hipSuccess);

    hipDeviceptr_t Ad{};
    hipDeviceptr_t Bd{};

    REQUIRE(hipMalloc((void**)&Ad, len * sz) == hipSuccess);
    REQUIRE(hipMalloc((void**)&Bd, len * sz) == hipSuccess);

    REQUIRE(hipMemcpyHtoD(Ad, data(A), len * sz) == hipSuccess);
    REQUIRE(hipMemcpyHtoD(Bd, data(B), len * sz) == hipSuccess);

    hipModule_t Module;
    hipFunction_t Function;
    REQUIRE(hipModuleLoad(&Module, fileName) == hipSuccess);
    REQUIRE(hipModuleGetFunction(&Function, Module, kernel_name) == hipSuccess);

    hipStream_t stream;
    REQUIRE(hipStreamCreate(&stream) == hipSuccess);

    hipDeviceptr_t _Ad{}; size_t _Ad_sz{};
    hipDeviceptr_t _Bd{}; size_t _Bd_sz{};

    REQUIRE(hipModuleGetGlobal(&_Ad, &_Ad_sz, Module, "a") == hipSuccess);
    REQUIRE(hipModuleGetGlobal(&_Bd, &_Bd_sz, Module, "b") == hipSuccess);

    // REQUIRE(_Ad_sz == sizeof(Ad));
    // REQUIRE(_Bd_sz == sizeof(Bd));

    REQUIRE(hipMemcpyToSymbol(_Ad, &Ad, sizeof(Ad)) == hipSuccess);
    REQUIRE(hipMemcpyToSymbol(_Bd, &Bd, sizeof(Bd)) == hipSuccess);

    void* args{};
    size_t size{};
    void* config[]{
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size,
        HIP_LAUNCH_PARAM_END};

    REQUIRE(hipModuleLaunchKernel(
        Function,
        1,
        1,
        1,
        len,
        1,
        1,
        0,
        stream,
        nullptr,
        (void**)&config) == hipSuccess);

    REQUIRE(hipStreamDestroy(stream) == hipSuccess);

    REQUIRE(hipMemcpyDtoH(data(B), Bd, sz) == hipSuccess);

    REQUIRE(equal(cbegin(A), cend(A), cbegin(B)));

	REQUIRE(hipCtxDestroy(context) == hipSuccess);
}