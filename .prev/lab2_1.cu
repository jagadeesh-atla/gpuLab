/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void blockdetails(void)
{
        printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
        printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
        printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
        int nElem = 3;
        dim3 block(256, 3, 1);
        dim3 grid((nElem + block.x - 1)/block.x, (nElem + block.y - 1)/block.y, (nElem + block.z - 1)/block.z);
        printf("blockDim:(%d, %d, %d)\n", block.x, block.y, block.z);
        printf("gridDim:(%d, %d, %d)\n", grid.x, grid.y, grid.z);
        blockdetails<<<grid, block>>>();
        cudaDeviceReset();
        return 0;
}
