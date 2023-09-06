/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MatrixMultKernel(float* A, float* B, float* C, int n){
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if ( (i < n) && (j < n) ) {
		float Pvalue=0;
		for (int k=0; k<n; k++)
			Pvalue += A[i*n + k]*B[k*n + j];
		C[i*n + j] = Pvalue;
	}	
}

__global__ void TransposeKernel(float* A, float* B, int n) {	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	if ( (i < n) && (j < n) ) {
		B[i*n + j] = A[j*n + i];
	}	
	
}

void initializeMatrix(float* A, int n) {
	for (int i=0; i<n; i++) {
		A[i] = i;
	}
}

void displayMatrix(float* A, int n) {
	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++)
			printf("%f ", A[i*n + j]);
		printf("\n");
	}
}

int main() {

	int n=4;
	int size=n*n;
	float *h_A, *h_B, *h_AT, *h_BT, *h_C;
	h_A = (float *)malloc(size*sizeof(float));
	h_B = (float *)malloc(size*sizeof(float));
	h_C = (float *)malloc(size*sizeof(float));
	h_AT = (float*)malloc(size*sizeof(float));
	h_BT = (float*)malloc(size*sizeof(float));

	float *d_A, *d_B, *d_C, *d_AT, *d_BT;
	cudaMalloc((void **)&d_A, size*sizeof(float));
	cudaMalloc((void **)&d_B, size*sizeof(float));
	cudaMalloc((void **)&d_C, size*sizeof(float));
	cudaMalloc((void**)&d_AT, size*sizeof(float));
	cudaMalloc((void**)&d_BT, size*sizeof(float));
	
	initializeMatrix(h_A, size);
	initializeMatrix(h_B, size);
	displayMatrix(h_A, n);
	
	cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 grid(2, 2, 1);
	dim3 block(2, 2, 1);

	MatrixMultKernel<<<grid, block>>>(d_A, d_B, d_C, n);
	
	TransposeKernel<<<grid, block>>>(d_A, d_AT, n);
	TransposeKernel<<<grid, block>>>(d_B, d_BT, n);

	cudaMemcpy(h_AT, d_AT, size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_BT, d_BT, size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("AT :-\n");
	displayMatrix(h_AT, n);

	printf("A*B :-\n");
	displayMatrix(h_C, n);
	
	MatrixMultKernel<<<grid, block>>>(d_AT, d_BT, d_C, n);
	cudaMemcpy(h_C, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
	printf("AT*BT :-\n");
	displayMatrix(h_C, n);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_AT);
	cudaFree(d_BT);
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_AT);
	free(h_BT);
	
	cudaDeviceReset();
	return 0;
}
