/* 21JE0192 - A JAGADEESH */
/**
 * Element wise sum of vector
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 10

__global__ void vecAddGPU(int *a, int *b, int *c) {
  int tid = threadIdx.x;

  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main() {
  int a[N], b[N], c[N];

  int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void **)&dev_a, N * sizeof(int));
  cudaMalloc((void **)&dev_b, N * sizeof(int));
  cudaMalloc((void **)&dev_c, N * sizeof(int));

  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = i * i;
  }

  cudaError_t err =
      cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  vecAddGPU<<<1, N>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) printf("%d + %d = %d\n", a[i], b[i], c[i]);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
