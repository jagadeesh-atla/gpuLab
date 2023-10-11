#include <cuda_runtime.h>
#include <stdio.h>

#include "operations.h"

#define THREADS_PER_BLOCK 16

__global__ void MulKernel(float *MatC, float *MatA, float *MatB, int Width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < Width && Col < Width) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += MatA[Row * Width + k] * MatB[k * Width + Col];
    }
    MatC[Row * Width + Col] = Pvalue;
  }
}

// grid 1D block 1D
__global__ void sumMatrix(float *MatC, float *MatA, float *MatB, int nx,
                          int ny) {
  unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);

  if (ix < nx) {
    for (int iy = 0; iy < ny; ++iy) {
      int idx = iy * nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }
}

void AddMatrix(float *out, float *in1, float *in2, int nx, int ny) {
  float *d_in1, *d_in2, *d_out;

  cudaMalloc((void **)&d_in1, nx * ny * sizeof(float));
  cudaMalloc((void **)&d_in2, nx * ny * sizeof(float));
  cudaMalloc((void **)&d_out, nx * ny * sizeof(float));

  cudaMemcpy(d_in1, in1, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_in2, in2, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

  sumMatrix<<<nx * ny / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_out, d_in1,
                                                                d_in1, nx, ny);

  cudaMemcpy(out, d_out, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}

void MulMatrix(float *out, float *in1, float *in2, int n) {
  float *d_in1, *d_in2, *d_out;

  cudaMalloc((void **)&d_in1, n * n * sizeof(float));
  cudaMalloc((void **)&d_in2, n * n * sizeof(float));
  cudaMalloc((void **)&d_out, n * n * sizeof(float));

  cudaMemcpy(d_in1, in1, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_in2, in2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  MulKernel<<<n * n / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_out, d_in1,
                                                              d_in1, n);
  cudaMemcpy(out, d_out, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in1);
  cudaFree(d_in2);
  cudaFree(d_out);
}
