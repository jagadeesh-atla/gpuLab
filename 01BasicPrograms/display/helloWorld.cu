/* 21JE0192 - A JAGADEESH */
/**
 *
 * Display Hello world on terminal from CPU & GPU threadaCUDA Sample Program
 *
 */
#include <stdio.h>

// kernel function
__global__ void helloFromGPU() { printf("Hello World from GPU!\n"); }

int main() {
  printf("Hello World from CPU!\n");

  // call kernal with 5 threads
  helloFromGPU<<<1, 10>>>();
  cudaDeviceReset();

  return 0;
}

