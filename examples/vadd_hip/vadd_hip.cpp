/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>

__global__
void vadd_hip(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

using namespace std;

int main()
{
    try{
        int sizeElements = 1000000;
        size_t sizeBytes = sizeElements * sizeof(float);
        bool pass = true;

        // Allocate host memory
        float* A_h = (float*)malloc(sizeBytes);
        float* B_h = (float*)malloc(sizeBytes);
        float* C_h = (float*)malloc(sizeBytes);

        // Allocate device memory:
        float *A_d, *B_d, *C_d;
        hipMalloc((void**)&A_d, sizeBytes);
        hipMalloc((void**)&B_d, sizeBytes);
        hipMalloc((void**)&C_d, sizeBytes);

        // Initialize host memory
        for (int i = 0; i < sizeElements; i++) {
            A_h[i] = 1.618f * i;
            B_h[i] = 3.142f * i;
        }

        // H2D Copy
        hipMemcpy(A_d, A_h, sizeBytes, hipMemcpyHostToDevice);
        hipMemcpy(B_d, B_h, sizeBytes, hipMemcpyHostToDevice);

        // Launch kernel onto default accelerator
        int blockSize = 256;                                      // pick arbitrary block size
        int blocks = (sizeElements + blockSize - 1) / blockSize;  // round up to launch enough blocks
        hipLaunchKernelGGL(vadd_hip, dim3(blocks), dim3(blockSize), 0, 0, A_d, B_d, C_d, sizeElements);

        // D2H Copy
        hipMemcpy(C_h, C_d, sizeBytes, hipMemcpyDeviceToHost);

        // Verify
        for (int i = 0; i < sizeElements; i++) {
            float ref = 1.618f * i + 3.142f * i;
            if (C_h[i] != ref) {
                printf("error:%d computed=%6.2f, reference=%6.2f\n", i, C_h[i], ref);
                pass = false;
            }
        };
        if (pass) printf("PASSED!\n");
    }
    catch (const exception& ex) {
        cerr << ex.what() << endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}