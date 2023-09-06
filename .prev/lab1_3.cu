/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {

        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
                printf("GPU not available\n");
        } else {
                printf("Detected %d device(s)\n", deviceCount);
        }

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("Warp Size: %d\n", deviceProp.warpSize);
        printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Max Memory Pitch: %d bytes\n", deviceProp.memPitch);
        return 0;
}
