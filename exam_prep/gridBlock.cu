#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel() {
  printf("threadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

  printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  const int nElem = 1024;
  dim3 block(1024);
  dim3 grid((nElem + block.x - 1) / block.x);

  block.x = 512;
  grid.x = (nElem + block.x - 1) / block.x;

  kernel<<<grid, block>>>();

  cudaDeviceReset();

  return 0;
}
