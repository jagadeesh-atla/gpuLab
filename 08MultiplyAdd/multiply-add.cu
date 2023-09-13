#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

__global__
void deviceFMAD(double a, double b, double* out){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid == 0) *out = a * a + b;
}

double hostFMAD(double a, double b) {
	return (a * a) + b;
}

int main() {
	double h_out, *d_out, d_value;

	double x = 1.323929;
	double y = -2.7436263;
	printf("x = %.20f\ny= %.20f\n\n", x, y);
	
	h_out = hostFMAD(x, y);

	cudaMalloc((void **)&d_out, sizeof(double));
	deviceFMAD<<<1, 32>>>(x, y, d_out);
	cudaMemcpy(&d_value, d_out, sizeof(double), cudaMemcpyDeviceToHost);

	double error = fabs(h_out - d_value);
	if (error == 0.0f) printf("Both are same\n");
	else printf("error of %.20f\n", error);
	
	printf("By Host = %.20f\nBy Device = %.20f\n", h_out, d_value);

	cudaFree(d_out);

	return 0;
}

