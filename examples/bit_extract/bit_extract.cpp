/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__
void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (auto i = offset; i < N; i += stride) {
#ifdef __HIP_PLATFORM_HCC__
        C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVCC__ or other path */
        C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
    }
}

using namespace std;

int main()
{
    try {
        uint32_t *A_d, *C_d;
        uint32_t *A_h, *C_h;
        size_t N = 1000000;
        size_t Nbytes = N * sizeof(uint32_t);

        int deviceId;
        CHECK(hipGetDevice(&deviceId));
        hipDeviceProp_t props;
        CHECK(hipGetDeviceProperties(&props, deviceId));
        printf("info: running on device #%d %s\n", deviceId, props.name);


        printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
        A_h = (uint32_t*)malloc(Nbytes);
        CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
        C_h = (uint32_t*)malloc(Nbytes);
        CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);

        for (auto i = 0u; i != N; i++) {
            A_h[i] = i;
        }

        printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
        CHECK(hipMalloc(&A_d, Nbytes));
        CHECK(hipMalloc(&C_d, Nbytes));

        printf("info: copy Host2Device\n");
        CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

        printf("info: launch 'bit_extract_kernel' \n");
        const unsigned blocks = 512;
        const unsigned threadsPerBlock = 256;
        hipLaunchKernelGGL(bit_extract_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

        printf("info: copy Device2Host\n");
        CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

        printf("info: check result\n");
        for (auto i = 0u; i != N; i++) {
            unsigned Agold = ((A_h[i] & 0xf00) >> 8);
            if (C_h[i] != Agold) {
                fprintf(stderr, "mismatch detected.\n");
                std::cerr << C_h[i] << ": " << C_h[i] << " =? " << Agold
                    << " (Ain=" << A_h[i] << ')' << std::endl;
                CHECK(hipErrorUnknown);
            }
        }
        printf("PASSED!\n");
    }
    catch (const exception& ex) {
        cerr << ex.what() << endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}