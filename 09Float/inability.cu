#include <cuda_runtime.h>

#include <stdio.h>

#define VALUE 121.1
#define VALUE_STR "121.1"

#define yrn(x, y) (x == y) ? "yes" : "no"

__global__
void Kernal(float *flt, double *dbl){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0) {
		*flt = VALUE;
		*dbl = VALUE;
	}
}

int main(){
	float h_flt = VALUE;
	double h_dbl = VALUE;
	
	float *d_flt, d_flt_value;
	double *d_dbl, d_dbl_value;

	cudaMalloc((void **)&d_flt, sizeof(float));
	cudaMalloc((void **)&d_dbl, sizeof(double));

	Kernal<<<1, 32>>> (d_flt, d_dbl);
	cudaMemcpy(&d_flt_value, d_flt, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&d_dbl_value, d_dbl, sizeof(double), cudaMemcpyDeviceToHost);

	printf("HOST\n");
	printf("Single-precision of "VALUE_STR" is %.20f\n", h_flt);
	printf("Double-precision of "VALUE_STR" is %.20f\n", h_dbl);

	printf("\n\n");

	printf("DEVICE\n");
	printf("Single-precision of "VALUE_STR" is %.20f\n", d_flt_value);
	printf("Double-precision of "VALUE_STR" is %.20f\n", d_dbl_value);

	printf("\n\n");
	printf("Single-precision same? %s\n", yrn(h_flt, d_flt_value));
	printf("Double-precision same? %s\n", yrn(h_dbl, d_dbl_value));

	return 0;
}

