#include <cuda_runtime.h>

#include <iostream>

#define N 10

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void VectorAdd(int* a, int* b, int* c) {
  int i = threadIdx.x;
  if (i < N) c[i] = a[i] + b[i];
}

int main() {
  int* a = new int[N];
  int* b = new int[N];
  int* c = new int[N];

  int *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, N * sizeof(int));
  cudaMalloc(&dev_b, N * sizeof(int));
  cudaMalloc(&dev_c, N * sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  VectorAdd<<<1, N>>>(dev_a, dev_b, dev_c);

  CHECK_CUDA_ERROR(
      cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) printf("%d + %d = %d\n", a[i], b[i], c[i]);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
