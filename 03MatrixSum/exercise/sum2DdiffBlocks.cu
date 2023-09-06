/* 21JE0192 - A JAGADEESH */
/**
 * Write a CUDA program to demonstrate
 * 1. Allocate Device Memory
 * 2. Transfer Data(Matrices A and B) from host to device
 * 3. Sum two matrices using 2D grid with different block sizes
 * 4. Transfer result(Matrix C) from device to host
 * 5. Print the result in matrix format
 * 6. Show the effect of block size and grid size in terms of total run time.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// #include <chrono>
#include <time.h>

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
  int nx = 1 << 12;
  int ny = 1 << 12;
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

  int blockSizes[] = {2, 4, 8, 16, 32, 64};
  int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
  //   int gridSize = (nx * ny + blockSizes[0] - 1) / blockSizes[0];

  printf("Matrix size: nx %d ny %d\n", nx, ny);

  clock_t start_time, end_time;
  double duration;

  for (int i = 0; i < numBlockSizes; ++i) {
    dim3 block(blockSizes[i], blockSizes[i]);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_time = clock();

    sumMatrix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

    cudaDeviceSynchronize();

    end_time = clock();
    duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Block Size: %d x %d, Grid Size: %d x %d, Time: %.10lf seconds\n",
           blockSizes[i], blockSizes[i], grid.x, grid.y, duration);
  }

  cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  //   printf("Matrix A:\n");
  //   displayMatrix(h_A, nx, ny);

  //   printf("Matrix B:\n");
  //   displayMatrix(h_B, nx, ny);

  //   printf("Matrix C (Sum of A and B):\n");
  //   displayMatrix(h_C, nx, ny);

  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
