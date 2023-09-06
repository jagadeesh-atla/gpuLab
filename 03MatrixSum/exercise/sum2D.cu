/* 21JE0192 - A JAGADEESH */
/**
 * Write a CUDA program to demonstrate the followings
 * 1. Allocate Device Memory
 * 2. Transfer Data(Matrices A and B) from host to device
 * 3. Sum two matrices using 2D grid
 * 4. Transfer result(Matrix C) from device to host
 * 5. Print the result in matrix format
 */

#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const int size) {
  for (int i = 0; i < size; ++i) {
    ip[i] = i;
  }
}

void displayMatrix(float *A, int nx, int ny) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      printf(" %f ", A[i * ny + j]);
    }
    printf("\n");
  }
}

__global__ void sumMatrix(float *MatA, float *MatB, float *MatC, int nx,
                          int ny) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix < nx && iy < ny) {
    int idx = ix * ny + iy;
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}

int main() {
  int nx = 4;
  int ny = 5;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  float *h_A, *h_B, *h_C;

  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);

  initialData(h_A, nxy);
  initialData(h_B, nxy);

  float *d_MatA, *d_MatB, *d_MatC;

  cudaMalloc((void **)&d_MatA, nBytes);
  cudaMalloc((void **)&d_MatB, nBytes);
  cudaMalloc((void **)&d_MatC, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  // dim3 block(1, 1);
  // dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  dim3 block(nx, ny);
  dim3 grid(1, 1);

  sumMatrix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

  cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  printf("Matrix A:\n");
  //   displayMatrix(h_A, nx, ny);

  printf("Matrix B:\n");
  //   displayMatrix(h_B, nx, ny);

  printf("Matrix C (Sum of A and B):\n");
  displayMatrix(h_C, nx, ny);

  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
