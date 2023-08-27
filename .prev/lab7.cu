#include "lab7.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "stdlib.h"

__global__ void basic_sum_reduce_kernel(float *dev_a, float *dev_b) {
	__shared__ float partialSum[BD];
  int tid = threadIdx.x;
	partialSum[tid] = dev_a[blockIdx.x * blockDim.x + tid];
	__syncthreads();
	for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if (tid % (2 * stride) == 0) {
			partialSum[tid] += partialSum[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0) {
		dev_b[blockIdx.x] = partialSum[0];
	}
}

struct Result basic_sum_reduce(float *arr, int n) {
  struct Result ans;
	float *dev_a, *dev_b, *b;
	int arr_size = n * sizeof(float);

	b = (float *)malloc(arr_size);
	dim3 block(BD, 1, 1);
	dim3 grid((block.x + n - 1) / block.x, 1, 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start, 0));

	CHECK(cudaMalloc((void**)&dev_a, arr_size));
	CHECK(cudaMalloc((void**)&dev_b, arr_size));
	CHECK(cudaMemcpy(dev_a, arr, arr_size, cudaMemcpyHostToDevice));

	basic_sum_reduce_kernel<<<grid, block>>>(dev_a, dev_b);
  basic_sum_reduce_kernel<<<1, block>>>(dev_b, dev_b);

	CHECK(cudaMemcpy(b, dev_b, arr_size, cudaMemcpyDeviceToHost));


	CHECK(cudaFree(dev_a));
	CHECK(cudaFree(dev_b));

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&ans.time, start, stop));
  CHECK(cudaDeviceReset());

  ans.ans = b[0];

	free(b);

	return ans;
}

__device__ void warp_reduce(volatile float *shared_sum, int tid) {
  shared_sum[tid] += shared_sum[tid + 32];
  shared_sum[tid] += shared_sum[tid + 16];
  shared_sum[tid] += shared_sum[tid + 8];
  shared_sum[tid] += shared_sum[tid + 4];
  shared_sum[tid] += shared_sum[tid + 2];
  shared_sum[tid] += shared_sum[tid + 1];
}

__global__ void improved_sum_reduce_kernel(float *dev_a, float *dev_b) {
	__shared__ float partialSum[BD];
  int tid = threadIdx.x;
	partialSum[tid] = dev_a[blockIdx.x * blockDim.x + tid];
	__syncthreads();
	for(unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		if (tid < stride) {
			partialSum[tid] += partialSum[tid + stride];
		}
		__syncthreads();
	}

  // Unroll last reduce
	if (tid < 32) {
    warp_reduce(partialSum, tid);
	}

  if (tid == 0) {
		dev_b[blockIdx.x] = partialSum[0];
  }
}

struct Result improved_sum_reduce(float *arr, int n) {
  struct Result ans;
	float *dev_a, *dev_b, *b;
	int arr_size = n * sizeof(float);

	b = (float *)malloc(arr_size);
	dim3 block(BD, 1, 1);
	dim3 grid((block.x + n - 1) / block.x, 1, 1);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start, 0));

	CHECK(cudaMalloc((void**)&dev_a, arr_size));
	CHECK(cudaMalloc((void**)&dev_b, arr_size));
	CHECK(cudaMemcpy(dev_a, arr, arr_size, cudaMemcpyHostToDevice));

	improved_sum_reduce_kernel<<<grid, block>>>(dev_a, dev_b);
  improved_sum_reduce_kernel<<<1, block>>>(dev_b, dev_b);

	CHECK(cudaMemcpy(b, dev_b, arr_size, cudaMemcpyDeviceToHost));


	CHECK(cudaFree(dev_a));
	CHECK(cudaFree(dev_b));

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaEventElapsedTime(&ans.time, start, stop));
  CHECK(cudaDeviceReset());

  ans.ans = b[0];

	free(b);

	return ans;
}


