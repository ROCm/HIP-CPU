/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */

#include "hip/hip_runtime.h"
#include <iostream>
#define NUM 1000000

#define HIP_CHECK(status)                                                                          \
    if (status != hipSuccess) {                                                                    \
        std::cout << "Got Status: " << status << " at Line: " << __LINE__ << std::endl;            \
        exit(0);                                                                                   \
    }

// Device (Kernel) function
__global__
void multiply(float* C, float* A, float* B, int N) {
    int tx = blockDim.x*blockIdx.x+threadIdx.x;

    if (tx < N){
	C[tx] = A[tx] * B[tx];
    }
}

// CPU implementation
void multiplyCPU(float* C, float* A, float* B, int N) {
    for(auto i = 0; i != N; ++i) {
        C[i] = A[i] * B[i];
    }
}

void launchKernel(float* C, float* A, float* B, bool manual) {
     hipDeviceProp_t devProp;
     HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

     hipEvent_t start, stop;
     HIP_CHECK(hipEventCreate(&start));
     HIP_CHECK(hipEventCreate(&stop));
     float eventMs = 1.0f;
     const unsigned threadsperblock = 32;
     const unsigned blocks = (NUM/threadsperblock)+1;

     uint32_t mingridSize = 0;
     uint32_t gridSize = 0;
     uint32_t blockSize = 0;

     if (!manual){
	blockSize = threadsperblock;
	gridSize  = blocks;
	std::cout << std::endl << "Manual Configuration with block size " << blockSize << std::endl;
     }
     else{
	HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&mingridSize, &blockSize, multiply, 0, 0));
	std::cout << std::endl << "Automatic Configuation based on hipOccupancyMaxPotentialBlockSize " << std::endl;
	std::cout << "Suggested blocksize is " << blockSize << ", Minimum gridsize is " << mingridSize << std::endl;
	gridSize = (NUM/blockSize)+1;
     }

     // Record the start event
     HIP_CHECK(hipEventRecord(start, NULL));

     // Launching the Kernel from Host
     hipLaunchKernelGGL(multiply, dim3(gridSize), dim3(blockSize), 0, 0, C, A, B, NUM);

     // Record the stop event
     HIP_CHECK(hipEventRecord(stop, NULL));
     HIP_CHECK(hipEventSynchronize(stop));

     HIP_CHECK(hipEventElapsedTime(&eventMs, start, stop));
     printf("kernel Execution time = %6.3fms\n", eventMs);

     //Calculate Occupancy
     int32_t numBlock = 0;
     HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, multiply, blockSize, 0));

     if(devProp.maxThreadsPerMultiProcessor){
	std::cout << "Theoretical Occupancy is " << (double)numBlock / devProp.maxThreadsPerMultiProcessor * 100 << "%" << std::endl;
     }
}

int main() {
     float *A, *B, *C0, *C1, *cpuC;
     float *Ad, *Bd, *C0d, *C1d;
     int errors=0;
     int i;

     // initialize the input data
     A  = (float *)malloc(NUM * sizeof(float));
     B  = (float *)malloc(NUM * sizeof(float));
     C0 = (float *)malloc(NUM * sizeof(float));
     C1 = (float *)malloc(NUM * sizeof(float));
     cpuC = (float *)malloc(NUM * sizeof(float));

     for(i=0; i< NUM; i++){
	     A[i] = static_cast<float>(i);
	     B[i] = static_cast<float>(i);
     }

     // allocate the memory on the device side
     HIP_CHECK(hipMalloc(&Ad, NUM * sizeof(float)));
     HIP_CHECK(hipMalloc(&Bd, NUM * sizeof(float)));
     HIP_CHECK(hipMalloc(&C0d, NUM * sizeof(float)));
     HIP_CHECK(hipMalloc(&C1d, NUM * sizeof(float)));

     // Memory transfer from host to device
     HIP_CHECK(hipMemcpy(Ad,A,NUM * sizeof(float), hipMemcpyHostToDevice));
     HIP_CHECK(hipMemcpy(Bd,B,NUM * sizeof(float), hipMemcpyHostToDevice));

     //Kernel launch with manual/default block size
     launchKernel(C0d, Ad, Bd, 1);

     //Kernel launch with the block size suggested by hipOccupancyMaxPotentialBlockSize
     launchKernel(C1d, Ad, Bd, 0);

     // Memory transfer from device to host
     HIP_CHECK(hipMemcpy(C0,C0d, NUM * sizeof(float), hipMemcpyDeviceToHost));
     HIP_CHECK(hipMemcpy(C1,C1d, NUM * sizeof(float), hipMemcpyDeviceToHost));

     // CPU computation
     multiplyCPU(cpuC, A, B, NUM);

     //verify the results
     double eps = 1.0E-6;

       for (i = 0; i < NUM; i++) {
	  if (std::abs(C0[i] - cpuC[i]) > eps) {
		  errors++;
	}
     }

     if (errors != 0){
	printf("\nManual Test FAILED: %d errors\n", errors);
	errors=0;
     } else {
	printf("\nManual Test PASSED!\n");
     }

     for (i = 0; i < NUM; i++) {
	  if (std::abs(C1[i] - cpuC[i]) > eps) {
		  errors++;
	}
     }

     if (errors != 0){
	printf("\n Automatic Test FAILED: %d errors\n", errors);
     } else {
	printf("\nAutomatic Test PASSED!\n");
     }

     HIP_CHECK(hipFree(Ad));
     HIP_CHECK(hipFree(Bd));
     HIP_CHECK(hipFree(C0d));
     HIP_CHECK(hipFree(C1d));

     free(A);
     free(B);
     free(C0);
     free(C1);
     free(cpuC);
}