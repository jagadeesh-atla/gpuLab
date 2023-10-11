#include <stdio.h>
#include <stdlib.h>

#include "operations.h"

const int N = 4;

void initializeMatrix(float* in) {
  for (int i = 0; i < N * N; ++i) {
    in[i] = ((float)rand() / (float)(RAND_MAX));
  }
}

void displayMatrix(float* A) {
  int idx, nx = N, ny = N;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      idx = i * ny + j;
      printf(" %f ", A[idx]);
    }
    printf("\n");
  }

  return;
}

int main() {
  float* in1 = (float*)calloc(N * N, sizeof(float));
  float* in2 = (float*)calloc(N * N, sizeof(float));
  float* out = (float*)calloc(N * N, sizeof(float));
  float* out2 = (float*)calloc(N * N, sizeof(float));

  initializeMatrix(in1);
  initializeMatrix(in2);

  printf("Matrix A: \n");
  displayMatrix(in1);

  printf("Matrix B: \n");
  displayMatrix(in2);

  AddMatrix(out, in1, in2, N, N);
  MulMatrix(out2, out, out, N);

  printf("Matrix A + B: \n");
  displayMatrix(out);

  printf("Matrix (A + B) * (A + B): \n");
  displayMatrix(out2);

  free(in1);
  free(in2);
  free(out);
  free(out2);

  return 0;
}