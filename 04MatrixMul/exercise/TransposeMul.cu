/* 21JE0192 - A JAGADEESH */
/**
 * Write a CUDA program to demonstrate
 * 1. Allocate Device Memory
 * 2. Transfer Data(Matrices A and B) from host to device
 * 3. Find the transpose (TA and TB) of matrices A and B in parallel on GPU
 * 4. Find the product of A and B and TA and TB
 * 5. Transfer results from device to host
 * 6. Print the result matrices and their differences
 * 7. Show the effect of block size and grid size in terms of total run time
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

const int N = 1 << 10;

__global__ void TransposeMul(float* MatA, float* MatAT, int Width) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i < Width) && (j < Width)) {
    MatAT[i * Width + j] = MatA[j * Width + i];
  }
}

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

  float *h_A, *h_B, *h_C, *h_AT, *h_BT, *h_CT;

  h_A = (float*)malloc(nBytes);
  h_B = (float*)malloc(nBytes);
  h_C = (float*)malloc(nBytes);
  h_AT = (float*)malloc(nBytes);
  h_BT = (float*)malloc(nBytes);
  h_CT = (float*)malloc(nBytes);

  initialData(h_A, nxy);
  initialData(h_B, nxy);

  float *d_MatA, *d_MatB, *d_MatC, *d_MatAT, *d_MatBT, *d_MatCT;
  cudaMalloc((void**)&d_MatA, nBytes);
  cudaMalloc((void**)&d_MatB, nBytes);
  cudaMalloc((void**)&d_MatC, nBytes);
  cudaMalloc((void**)&d_MatAT, nBytes);
  cudaMalloc((void**)&d_MatBT, nBytes);
  cudaMalloc((void**)&d_MatCT, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  int blockSizes[] = {2, 4, 8, 16, 32, 64};
  int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

  printf("Matrix size: nx %d ny %d\n", nx, ny);

  clock_t start_time, end_time;
  double duration;

  for (int i = 0; i < numBlockSizes; ++i) {
    dim3 block(blockSizes[i], blockSizes[i]);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    start_time = clock();

    TransposeMul<<<grid, block>>>(d_MatA, d_MatAT, Width);
    TransposeMul<<<grid, block>>>(d_MatB, d_MatBT, Width);
    MatrixMul<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Width);
    MatrixMul<<<grid, block>>>(d_MatAT, d_MatBT, d_MatCT, Width);

    cudaDeviceSynchronize();

    end_time = clock();
    duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Block Size: %d x %d, Grid Size: %d x %d, Time: %.10lf seconds\n",
           blockSizes[i], blockSizes[i], grid.x, grid.y, duration);
  }

  cudaMemcpy(h_AT, d_MatAT, nBytes, cudaMemcpyDeviceToHost);

  cudaMemcpy(h_BT, d_MatBT, nBytes, cudaMemcpyDeviceToHost);

  cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  cudaMemcpy(h_CT, d_MatCT, nBytes, cudaMemcpyDeviceToHost);

  //   printf("Matrix A is \n");
  //   displayMatrix(h_A, nx, ny);

  //   printf("Matrix B is \n");
  //   displayMatrix(h_B, nx, ny);

  //   printf("Product of Matrix A and B is \n");
  //   displayMatrix(h_C, nx, ny);

  //   printf("Matrix AT is \n");
  //   displayMatrix(h_AT, nx, ny);

  //   printf("Matrix BT is \n");
  //   displayMatrix(h_BT, nx, ny);

  //   printf("Product of Matrix AT and BT is \n");
  //   displayMatrix(h_CT, nx, ny);

  // printf("Block Size: %d x %d, Grid Size: %d x %d, Time: %.10lf seconds\n",
  //        block.x, block.y, grid.x, grid.y, duration);

  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);
  cudaFree(d_MatAT);
  cudaFree(d_MatBT);
  cudaFree(d_MatCT);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_AT);
  free(h_BT);
  free(h_CT);

  cudaDeviceReset();

  return 0;
}