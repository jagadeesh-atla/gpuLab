/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>

#include "funcDef.h"

#define N 2

#define TILE_WIDTH 1

__global__ void MatrixMulKernel(float* MatA, float *MatB, float *MatC, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;

	float Pvalue = 0;
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
		Mds[ty][tx] = MatB[Row*Width + (ph*TILE_WIDTH + tx)];
		Nds[ty][tx] = MatB[(ph*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();
	
		for (int k = 0; k < TILE_WIDTH; ++k) 
		{Pvalue += Mds[ty][k] * Nds[k][tx];
			__syncthreads();}
		
	}
	MatC[(Row*Width) + Col] = Pvalue;
}

void displayMatrix(float *A, int nx, int ny) {
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

	printf("Matrix size: %d by %d\n", nx, ny);

	printf("Tile size: %d by %d\n", TILE_WIDTH, TILE_WIDTH);

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
	
	int bdimx = TILE_WIDTH;
	int bdimy = TILE_WIDTH;
	
	dim3 block(bdimx, bdimy);
	dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1) / block.y, 1);

	MatrixMulKernel<<<grid, block>>>(d_MatA, d_MatB, d_MatC, Width);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_MatC, nBytes, cudaMemcpyDeviceToHost);
	printf("Matrix A is = \n");
	displayMatrix(h_A, nx, ny);
	printf("Matrix B is = \n");
	displayMatrix(h_B, nx, ny);
	printf("Matrix C = A * B\n");
	displayMatrix(h_C, nx, ny);

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaDeviceReset();
	
	return 0;
}

