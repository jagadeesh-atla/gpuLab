#ifndef KERNEL_H
#define KERNEL_H

__device__ void matAdd(float *mat_A, float *mat_B, float *mat_C);

__global__ void matAddKernel(float *mat_A, float *mat_B, float *mat_C);

__global__ void squareMat(float *mat);

#endif