#include <cuda_runtime.h>
#include <stdio.h>

#define SIZE (100 * 1024 * 104)
#define HANDLE_ERROR(call)                                   \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s.%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code %d, reason: %s\n", error,        \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

void *big_block(int size) {
  unsigned char *data = (unsigned char *)malloc(size);
  unsigned char userBeg = 0, userEnd = 255;
  for (int i = 0; i < size; ++i)
    data[i] = rand() % ((userEnd - userBeg) + 1) + userBeg;
  return data;
}

// __global__ void histo_kernel(unsigned char *buffer, long size,
//                              unsigned int *histo) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x * gridDim.x;
//   while (i < size) {
//     {
//       atomicAdd(&(histo[buffer[i]]), 1);
//       i += stride;
//     }
//   }
// }

// sharedMemory
__global__ void histo_kernel(unsigned char *buffer, long size,
                             unsigned int *histo) {
  __shared__ unsigned int temp[256];
  temp[threadIdx.x] = 0;
  __syncthreads();

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd(&temp[buffer[i]], 1);
    i += offset;
  }

  __syncthreads();
  atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

int main() {
  unsigned char *buffer = (unsigned char *)big_block(SIZE);

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  unsigned char *dev_buffer;
  unsigned int *dev_histo;

  HANDLE_ERROR(cudaMalloc((void **)&dev_buffer, SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **)&dev_histo, 256 * sizeof(long)));
  HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

  int blocks = prop.multiProcessorCount;

  histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

  unsigned int histo[256];
  HANDLE_ERROR(
      cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  long histoCount = 0;
  for (int i = 0; i < 256; ++i) {
    histoCount += histo[i];
    printf("%c - %6.ld;\n", i, histo[i]);
  }

  printf("Elapsed time: %3.lf ms\n", elapsedTime);
  printf("Histo count: %ld", histoCount);

  for (int i = 0; i < SIZE; ++i) {
    histo[buffer[i]]--;
  }
  for (int i = 0; i < 256; ++i) {
    if (histo[i] != 0) {
      printf("Failure at %d!\n", i);
    }
  }

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  cudaFree(dev_histo);
  cudaFree(dev_buffer);
  free(buffer);

  return 0;
}
