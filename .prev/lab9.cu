/* 21JE0192 - A JAGADEESH */
#include "lab9.h"
#include "stdio.h"

void initialData(int *ip, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    ip[i] = rand() % 100;
  }
}

void print_arr(int *arr, const size_t size) {
  printf("[ ");
  for (size_t i = 0; i < size; i++) {
    printf("%d, ", arr[i]);
  }
  printf("]\n");
}

int cpu_sum(int *arr, const size_t len) {
  int sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += arr[i];
  }
  return sum;
}

int cpu_max(int *arr, const size_t len) {
  int m = 0;
  for (size_t i = 0; i < len; ++i) {
    if (m < arr[i]) {
      m = arr[i];
    }
  }
  return m;
}

__device__ void warp_reduce(volatile int *shared_sum, int tid) {
  shared_sum[tid] += shared_sum[tid + 32];
  shared_sum[tid] += shared_sum[tid + 16];
  shared_sum[tid] += shared_sum[tid + 8];
  shared_sum[tid] += shared_sum[tid + 4];
  shared_sum[tid] += shared_sum[tid + 2];
  shared_sum[tid] += shared_sum[tid + 1];
}

__global__ void improved_sum_reduce_kernel(int *dev_a, int *dev_b, const size_t len) {
	__shared__ int partialSum[BD];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  if (index < len) {
	  partialSum[tid] = dev_a[index];
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
}

// if b > a => b = a
// Else Nothing
__device__ void max_exchange(volatile int *a, volatile int *b) {
  if (*b > *a) {
    *b = *a;
  }
}

__device__ void max_reduce(volatile int *shared_sum, int tid) {
  max_exchange(&shared_sum[tid], &shared_sum[tid + 32]);
  max_exchange(&shared_sum[tid], &shared_sum[tid + 16]);
  max_exchange(&shared_sum[tid], &shared_sum[tid + 8]);
  max_exchange(&shared_sum[tid], &shared_sum[tid + 4]);
  max_exchange(&shared_sum[tid], &shared_sum[tid + 2]);
  max_exchange(&shared_sum[tid], &shared_sum[tid + 1]);
}

__global__ void max_array_kernel(int *dev_a, int *dev_b, const size_t len) {
	__shared__ int partialSum[BD];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  if (index < len) {
	  partialSum[tid] = dev_a[index];
	  __syncthreads();
	  for(unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
	  	if (tid < stride) {
        if (partialSum[tid] < partialSum[tid + stride]) {
          partialSum[tid] = partialSum[tid + stride];
        }
	  	}
	  	__syncthreads();
	  }

    	// Unroll last reduce
	  if (tid < 32) {
      		max_reduce(partialSum, tid);
	  }

    if (tid == 0) {
	  	dev_b[blockIdx.x] = partialSum[0];
    }
  }
}

void doubleStream(const size_t chunk, const size_t full_size) {
  int *host_arr, *host_sum, *host_max;
  int *dev_arr_stream1, *dev_arr_stream2, *dev_sum_stream1, *dev_max_stream2;
  float timer_stream1, timer_stream2;
  cudaStream_t stream1, stream2;
  cudaEvent_t start_stream1, stop_stream1, start_stream2, stop_stream2;

  //start the timers
  CHECK(cudaEventCreate(&start_stream1));
  CHECK(cudaEventCreate(&stop_stream1));
  CHECK(cudaEventCreate(&start_stream2));
  CHECK(cudaEventCreate(&stop_stream2));

  CHECK(cudaStreamCreate(&stream1));
  CHECK(cudaStreamCreate(&stream2));

  CHECK(cudaEventRecord(start_stream1, stream1));
  CHECK(cudaEventRecord(start_stream2, stream2));

  CHECK(cudaMalloc((void **)&dev_arr_stream1, chunk * sizeof(int)));
  CHECK(cudaMalloc((void **)&dev_arr_stream2, chunk * sizeof(int)));
  CHECK(cudaMalloc((void **)&dev_sum_stream1, chunk * sizeof(int)));
  CHECK(cudaMalloc((void **)&dev_max_stream2, chunk * sizeof(int)));

  CHECK(cudaHostAlloc((void **)&host_arr, full_size * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void **)&host_sum, full_size * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void **)&host_max, full_size * sizeof(int),
                      cudaHostAllocDefault));

  initialData(host_arr, full_size);

  for (size_t i = 0; i < full_size; i += chunk) {
    CHECK(cudaMemcpyAsync(dev_arr_stream1, host_arr + i, chunk * sizeof(int),
                          cudaMemcpyHostToDevice, stream1));
    CHECK(cudaMemcpyAsync(dev_arr_stream2, host_arr + i, chunk * sizeof(int),
                          cudaMemcpyHostToDevice, stream2));

    dim3 block(BD);
    dim3 grid((chunk + block.x - 1) / block.x);

    improved_sum_reduce_kernel<<<grid, block, 0, stream1>>>(dev_arr_stream1, dev_sum_stream1, chunk);
    max_array_kernel<<<grid, block, 0, stream2>>>(dev_arr_stream2, dev_max_stream2, chunk);
    improved_sum_reduce_kernel<<<1, block, 0, stream1>>>(dev_sum_stream1, dev_sum_stream1, chunk);
    max_array_kernel<<<1, block, 0, stream2>>>(dev_max_stream2, dev_max_stream2, chunk);

    CHECK(cudaMemcpyAsync(host_sum + i, dev_sum_stream1, chunk * sizeof(int),
                          cudaMemcpyDeviceToHost, stream1));
    CHECK(cudaMemcpyAsync(host_max + i, dev_max_stream2, chunk * sizeof(int),
                          cudaMemcpyDeviceToHost, stream2));
  }

  CHECK(cudaStreamSynchronize(stream1));
  CHECK(cudaFree(dev_arr_stream1));
  CHECK(cudaFree(dev_sum_stream1));
  CHECK(cudaEventRecord(stop_stream1, stream1));

  CHECK(cudaStreamSynchronize(stream2));
  CHECK(cudaFree(dev_arr_stream2));
  CHECK(cudaFree(dev_max_stream2));
  CHECK(cudaEventRecord(stop_stream2, stream2));

  CHECK(cudaEventSynchronize(stop_stream1));
  CHECK(cudaEventSynchronize(stop_stream2));

  int sum = 0;
  for(size_t i = 0; i < full_size; i += chunk)  {
    sum += host_sum[i];
  }

  int max = 0;
  for(size_t i = 0; i < full_size; i += chunk)  {
    if (max < host_max[i]) {
      max = host_max[i];
    }
  }


  if (cpu_max(host_arr, full_size) != max) {
    printf("Incorrect Maximum");
  }

  if (cpu_sum(host_arr, full_size) != sum) {
    printf("Incorrect Sum");
  }

  CHECK(cudaEventElapsedTime(&timer_stream2, start_stream2, stop_stream2));
  CHECK(cudaEventElapsedTime(&timer_stream1, start_stream1, stop_stream1));

  printf("Execution Time of stream with ID %d: %f\n", stream1, timer_stream1);
  printf("Execution Time of stream with ID %d: %f\n", stream2, timer_stream2);

  CHECK(cudaFreeHost(host_arr));
  CHECK(cudaFreeHost(host_sum));
  CHECK(cudaFreeHost(host_max));
  CHECK(cudaStreamDestroy(stream1));
  CHECK(cudaStreamDestroy(stream2));

}


