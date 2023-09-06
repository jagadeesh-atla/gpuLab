/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

void initializeMatrix(float* A, int m, int n){
	for(int i=0; i<m*n; i++)
		A[i] = i;
}

void displayMatrix(float* A, int m, int n) {
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			printf("%.0f ", A[i*n + j]);
		printf("\n");
	}
}

__global__ void sumMatrixKernel(float* A, float* B, float* C, int nx, int ny) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if ((i<nx) && (j<ny))
		C[i*ny + j] = A[i*ny + j] + B[i*ny + j];
}

int main() {
	int m=1024;
	int n=1024;
	int size = m*n;

	float *h_A, *h_B, *h_C;
	h_A = (float*)malloc(size*sizeof(float));
	h_B = (float*)malloc(size*sizeof(float));
	h_C = (float*)malloc(size*sizeof(float));

	initializeMatrix(h_A, m, n);
	initializeMatrix(h_B, m, n);	

	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, size*sizeof(float));
	cudaMalloc((void**)&d_B, size*sizeof(float));
	cudaMalloc((void**)&d_C, size*sizeof(float));

	cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 grid(256, 256, 1);
	dim3 block(4, 4, 1);
	
	printf("Adding 2 matrices of size 1024x1024\n\n");	
	sumMatrixKernel<<<grid, block>>>(d_A, d_B, d_C, m, n);
	cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
	//displayMatrix(h_C, m, n);
	
	struct timeval stop, start;
	grid.x = 256; grid.y = 256;
	block.x = 4; block.y = 4;

	gettimeofday(&start, NULL);
	sumMatrixKernel<<<grid, block>>>(d_A, d_B, d_C, m, n);
	gettimeofday(&stop, NULL);
	printf("Time taken for grid(256,256,1) and block(4,4,1) = %lu microsecs\n",
		(stop.tv_sec - start.tv_sec)*1000000 + stop.tv_usec - start.tv_usec);	
	cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
	
	grid.x = 4; grid.y = 4;
	block.x = 256; block.y = 256;

	gettimeofday(&start, NULL);
	sumMatrixKernel<<<grid, block>>>(d_A, d_B, d_C, m, n);
	gettimeofday(&stop, NULL);
	printf("Time taken for grid(4,4,1) and block(256,256,1) = %lu microsecs\n",
		(stop.tv_sec - start.tv_sec)*1000000 + stop.tv_usec - start.tv_usec);	
	cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
