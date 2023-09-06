/* 21JE0192 - A JAGADEESH */
/**
 *
 * Write a CUDA C program to display your name 10-10 times from CPU and GPU
 * respectively
 *
 */
#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello Jagadeesh from GPU!\n"); }

void helloFromCPU() {
  for (int i = 0; i < 10; ++i) {
    printf("Hello Jagadeesh from CPU!\n");
  }
}

int main() {
  helloFromCPU();

  helloFromGPU<<<1, 10>>>();
  cudaDeviceReset();

  return 0;
}
