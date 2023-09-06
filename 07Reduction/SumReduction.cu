#include <cuda_runtime.h>

#include <stdio.h>

#define N 100
#define BD 256

#define CHECK(call) \
{ \
	const cudaError_t err = call;\
	if (err != cudaSuccess) {\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", err,\
				cudaGetErrorString(err));\
		exit(1);\
	}\
}

__global__ void SumReduction(float *dev_a, float *dev_d) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	
	__shared__ float partialSum[BD];
	partialSum[t] = dev_a[i];

	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (t % (2 * stride) == 0 ) 
			partialSum[t] += partialSum[t+stride];
	}
	dev_d[0] = partialSum[0];
}

int main() {
	float a[N], b[N];
	float *dev_a, *dev_d;

	int bdimx = BD;
	float elapsedTime;

	dim3 block(bdimx);
	dim3 grid((N + block.x - 1)/block.x, 1, 1);

	cudaEvent_t start, stop;
	
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	
	printf("Array Size is %d\n", N);
	
	CHECK(cudaMalloc((void **)&dev_a, N * sizeof(float)));
	CHECK(cudaMalloc((void **)&dev_d, N * sizeof(float)));
	
	for (int i = 0; i < N; ++i) {	
		float x  = ((float)rand())/(float)(RAND_MAX);
		a[i] = x;
		if (N < 25) printf("%f ", x);
	}
	printf("\n");	

	CHECK(cudaEventRecord(start, 0));
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time for Memcpy: %8.6f ms\n", elapsedTime);

	CHECK(cudaEventRecord(start, 0));
	SumReduction<<<grid, block>>>(dev_a, dev_d);
	CHECK(cudaMemcpy(b, dev_d, N * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time for Sum: %8.6f ms\n", elapsedTime);

	printf("Sum = %f\n", b[0]);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_a);
	cudaFree(dev_d);

	return 0;
}

