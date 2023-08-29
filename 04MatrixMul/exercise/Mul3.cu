/**
 *Write a CUDA program to demonstrate the followings
 1. Allocate Device Memory
 2. Transfer Data(Matrices A, B and C) from host to device
 3. Find the Product of three matrices A*B*C using 2D grid
 4. Transfer result from device to host
 5. Print the result in matrix format
 */

#include <cuda_runtime.h>
#include <stdio.h>

const int N = 1 << 10;

__global__ void MatrixMul(float* MatA, float* MatB, float* MatC, int Width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < Width && Col < Width) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += MatA[Row * Width + k] * MatB[k * Width + Col];
    }
    MatC[Row * Width + Col] = Pvalue;
  }
}

void initialData(float* ip, const int size) {
  int i;
  for (i = 0; i < size; ++i) {
    ip[i] = ((float)rand() / (float)(RAND_MAX));
    // ip[i] = i;
  }
  return;
}

void displayMatrix(float* A, int nx, int ny) {
  int idx;
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
  int Width = N;
  int nx = Width;
  int ny = Width;

  int nxy = nx * ny;

  int nBytes = nxy * sizeof(float);

  printf("Matrix of size: nx %d ny %d\n", nx, ny);

  float *h_A, *h_B, *h_C, *h_D, *h_E;

  h_A = (float*)malloc(nBytes);
  h_B = (float*)malloc(nBytes);
  h_C = (float*)malloc(nBytes);
  h_D = (float*)malloc(nBytes);
  h_E = (float*)malloc(nBytes);

  initialData(h_A, nxy);
  initialData(h_B, nxy);
  initialData(h_C, nxy);

  float *d_MatA, *d_MatB, *d_MatC, *d_MatD, *d_MatE;
  cudaMalloc((void**)&d_MatA, nBytes);
  cudaMalloc((void**)&d_MatB, nBytes);
  cudaMalloc((void**)&d_MatC, nBytes);
  cudaMalloc((void**)&d_MatD, nBytes);
  cudaMalloc((void**)&d_MatE, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatC, h_C, nBytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

  MatrixMul<<<grid, block>>>(d_MatA, d_MatB, d_MatD, Width);
  cudaDeviceSynchronize();

  MatrixMul<<<grid, block>>>(d_MatD, d_MatC, d_MatE, Width);
  cudaDeviceSynchronize();

  cudaMemcpy(h_D, d_MatD, nBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_E, d_MatE, nBytes, cudaMemcpyDeviceToHost);

  //   printf("Matrix A is \n");
  //   displayMatrix(h_A, nx, ny);

  //   printf("Matrix B is \n");
  //   displayMatrix(h_B, nx, ny);

  //   printf("Matrix C is \n");
  //   displayMatrix(h_C, nx, ny);

  //   printf("Product of Matrix A, B and C is \n");
  //   displayMatrix(h_E, nx, ny);

  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);
  cudaFree(d_MatD);
  cudaFree(d_MatE);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_D);
  free(h_E);

  cudaDeviceReset();

  return 0;
}