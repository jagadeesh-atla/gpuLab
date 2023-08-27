#ifndef KERNEL_H
#define KERNEL_H

__device__ void transpose(float *mat_A, float *mat_B);
__global__ void transposeKernel (float *mat_A, float *mat_b);
__global__ void squareMat(float *mat);

#endif