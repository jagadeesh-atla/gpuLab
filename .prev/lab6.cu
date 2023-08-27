#include <cuda_runtime.h>
#include <stdio.h>
#include "lab6.h"

#define N 8
#define TILE_WIDTH 2


void initialData(float* ip, const int size) {
	int i;
	for(i = 0; i<size; i++)
		ip[i]=((float)rand()/(float)(RAND_MAX));
	return;
}

void displayMatrix(float* A, int nx, int ny){
	int idx;
	for (int i=0; i<nx; i++){
		for (int j=0; j<ny; j++){
			idx = i*ny + j;
			printf("%f ", A[idx]);
		}
		printf("\n");
	}
}

__global__ void TiledMatrixMulKernel(float* MatA, float* MatB, float* MatC, int width) {
        __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;
        int row = by*TILE_WIDTH + ty;
        int col = bx*TILE_WIDTH + tx;

        float Pvalue = 0;
        for (int ph=0; ph < width/TILE_WIDTH; ph++) {
                Mds[ty][tx] = MatA[row*width + ph*TILE_WIDTH + tx];
                Nds[ty][tx] = MatB[(ph*TILE_WIDTH + ty)*width + col];
                __syncthreads();
                for (int k=0; k<TILE_WIDTH; k++) {
                        Pvalue += Mds[ty][k]*Nds[k][tx];
                }
                __syncthreads();
        }
        MatC[row*width + col] = Pvalue;
}

__global__ void MatrixMulKernel(float* MatA, float* MatB, float* MatC, int width) {
        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;
        int row = by*TILE_WIDTH + ty;
        int col = bx*TILE_WIDTH + tx;

        float Pvalue = 0;
        for (int k=0; k<width; k++) {
            Pvalue += MatA[row*width + k] * MatB[k*width + col];
        }
        
        MatC[row*width + col] = Pvalue;
}

int main() {
	int width=N;
	int nx=width;
	int ny=width;
	int nxy = nx*ny;
	int nBytes = nxy*sizeof(float);

	printf("Tile size: %d x %d\n", TILE_WIDTH, TILE_WIDTH);

	float *h_A, *h_B, *h_C;
	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	h_C = (float*)malloc(nBytes);

	initialData(h_A, nxy);
	initialData(h_B, nxy);
	printf("Matrix A:\n");
        displayMatrix(h_A, nx, ny);
        printf("Matrix B:\n");
        displayMatrix(h_B, nx, ny);

	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, nBytes);
	cudaMalloc((void **)&d_B, nBytes);
	cudaMalloc((void **)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	int bdimx = TILE_WIDTH;
	int bdimy = TILE_WIDTH;

	dim3 block(bdimx, bdimy);
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, 1);
	
	TiledMatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, width);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
	printf("Multiplication with Tiles :-\n");
	displayMatrix(h_C, nx, ny);


	initialData(h_A, nxy);
        initialData(h_B, nxy);
	
	printf("Matrix A:\n");
        displayMatrix(h_A, nx, ny);
        printf("Matrix B:\n");
        displayMatrix(h_B, nx, ny);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	
	MatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, width);
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
	printf("Multiplication without Tiles :-\n");
	displayMatrix(h_C, nx, ny);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaDeviceReset();

	return 0;
}



























