#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 20)
#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s.%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code %d, reason: %s\n", error,        \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

__global__ void kernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    c[idx] = (a[idx] + b[idx]) / 2.0;
  }
}

int main() {
  cudaDeviceProp prop;
  int which;
  CHECK(cudaGetDevice(&which));
  CHECK(cudaGetDeviceProperties(&prop, which));

  if (!prop.deviceOverlap) {
    printf("Device will not handle overlap, so speed up from stream");
    return 0;
  }

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaStream_t stream;
  int *host_a, *host_b, *host_c;
  int *dev_a, *dev_b, *dev_c;

  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaStreamCreate(&stream));

  CHECK(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  CHECK(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  CHECK(cudaMalloc((void **)&dev_c, N * sizeof(int)));

  CHECK(cudaHostAlloc((void **)&host_a, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void **)&host_b, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));
  CHECK(cudaHostAlloc((void **)&host_c, FULL_DATA_SIZE * sizeof(int),
                      cudaHostAllocDefault));

  for (int i = 0; i < FULL_DATA_SIZE; ++i) {
    host_a[i] = rand();
    host_b[i] = rand();
  }

  CHECK(cudaEventRecord(start, 0));
  for (int i = 0; i < FULL_DATA_SIZE; i += N) {
    CHECK(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int),
                          cudaMemcpyHostToDevice, stream));

    kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);

    CHECK(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream));
  }

  CHECK(cudaStreamSynchronize(stream));

  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));

  CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  printf("Stream ID: %d, \nElapsed Time: %8.6f\n", stream, elapsedTime);

  CHECK(cudaFreeHost(host_a));
  CHECK(cudaFreeHost(host_b));
  CHECK(cudaFreeHost(host_c));

  CHECK(cudaFree(dev_a));
  CHECK(cudaFree(dev_b));
  CHECK(cudaFree(dev_c));

  CHECK(cudaStreamDestroy(stream));

  return 0;
}
