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
	float *h_A, *h_B, *h_C, *h_D, *h_E;
	h_A = (float *)malloc(size*sizeof(float));
	h_B = (float *)malloc(size*sizeof(float));
	h_C = (float *)malloc(size*sizeof(float));
	h_D = (float*)malloc(size*sizeof(float));
	h_E = (float*)malloc(size*sizeof(float));

	float *d_A, *d_B, *d_C, *d_D, *d_E;
	cudaMalloc((void **)&d_A, size*sizeof(float));
	cudaMalloc((void **)&d_B, size*sizeof(float));
	cudaMalloc((void **)&d_C, size*sizeof(float));
	cudaMalloc((void**)&d_D, size*sizeof(float));
	cudaMalloc((void**)&d_E, size*sizeof(float));
	
	initializeMatrix(h_A, size);
	initializeMatrix(h_B, size);
	initializeMatrix(h_C, size);
	printf("Matrix A, B, C:-\n");
	displayMatrix(h_A, n);
	
	cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 grid(2, 2, 1);
	dim3 block(2, 2, 1);

	MatrixMultKernel<<<grid, block>>>(d_A, d_B, d_D, n);
	MatrixMultKernel<<<grid, block>>>(d_D, d_C, d_E, n);
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_D, d_D, size*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_E, d_E, size*sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("A*B :-\n");
	displayMatrix(h_D, n);
	
	printf("A*B*C :-\n");
	displayMatrix(h_E, n);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
	cudaFree(d_E);
	
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_D);
	free(h_E);
	
	cudaDeviceReset();
	return 0;
}
