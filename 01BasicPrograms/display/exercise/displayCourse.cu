/* 21JE0192 - A JAGADEESH */
/**
 * Write a CUDA C program to display your 4 times Course Name, Name of
 * Experiment and Date from CPU and GPU respectively
 */

#include <stdio.h>

#define details \
  "Name: GPU Computing Lab\n\
Experiment : Display Course Details\n\
Date : Aug 09,2023\n"

__global__ void printFromGPU() { printf(details); }

void printFromCPU() {
  for (int i = 0; i < 4; ++i) printf(details);
}

int main() {
  printf("======CPU======\n");
  printFromCPU();

  printf("======GPU======\n");
  printFromGPU<<<1, 4>>>();
  cudaDeviceReset();

  return 0;
}
