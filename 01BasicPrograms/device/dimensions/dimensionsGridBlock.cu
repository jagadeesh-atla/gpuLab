/**
 * Display the dimensions number of threads in block and number of block in the
 * grid.
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
  printf("threadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);

  printf("blockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

  printf("blockDim: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);

  printf("gridDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  int nElem = 3;

  dim3 block(3);
  dim3 grid((nElem + block.x - 1) / block.x);

  printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
  printf("block: (%d, %d, %d)\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();
  cudaDeviceReset();

  return 0;
}
