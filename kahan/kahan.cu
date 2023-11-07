#include <cassert>
#include <cmath>
#include <iostream>

__global__ void kahanSum(float* data, int n, float* result) {
  float sum = 0.0f;
  float c = 0.0f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float y = data[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  atomicAdd(result, sum);
}

int main() {
  int n = 1000;  // Adjust this to your data size
  float* h_data = new float[n];
  float* d_data;
  float h_result = 0.0f;
  float* d_result;

  // Initialize and allocate memory on the GPU
  for (int i = 0; i < n; ++i) {
    h_data[i] = i;  // You can replace this with your data
  }
  cudaMalloc(&d_data, n * sizeof(float));
  cudaMalloc(&d_result, sizeof(float));

  // Copy data to the GPU
  cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel to compute the Kahan sum
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  kahanSum<<<gridSize, blockSize>>>(d_data, n, d_result);

  // Copy the result back to the CPU
  cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

  // Free allocated memory
  cudaFree(d_data);
  cudaFree(d_result);

  double s = 0.0f;
  for (int i = 0; i < n; ++i) s += h_data[i];

  assert(s == h_result);

  std::cout << "Kahan Sum: " << h_result << std::endl;

  delete[] h_data;

  return 0;
}
