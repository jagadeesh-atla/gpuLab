/**
 * distance b/w two vectors x = i^2, y = (2i + 1) and n = 1024. Also find
 * euclidean norms
 *
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#define N 1024

__global__ void distance(double *a, double *b, double *c) {
  int i = threadIdx.x;
  if (i < N) c[i] = (a[i] - b[i]) * (a[i] - b[i]);
}

__global__ void euclideanNorm(double *a, double *c) {
  int i = threadIdx.x;
  if (i < N) c[i] = a[i] * a[i];
}

int main() {
  double x[N], y[N], temp[N], dist = 0, xNorm = 0, yNorm = 0;

  for (int i = 1; i <= N; ++i) {
    x[i - 1] = i * i;
    y[i - 1] = 2 * i + 1;
  }

  double *dev_x, *dev_y, *dev_d, *dev_xN, *dev_yN;

  cudaMalloc((void **)&dev_x, N * sizeof(double));
  cudaMalloc((void **)&dev_y, N * sizeof(double));
  cudaMalloc((void **)&dev_d, N * sizeof(double));
  cudaMalloc((void **)&dev_xN, N * sizeof(double));
  cudaMalloc((void **)&dev_yN, N * sizeof(double));

  cudaMemcpy(dev_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, N * sizeof(double), cudaMemcpyHostToDevice);

  distance<<<1, N>>>(dev_x, dev_y, dev_d);

  cudaMemcpy(temp, dev_d, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) dist += temp[i];

  euclideanNorm<<<1, N>>>(dev_x, dev_xN);
  cudaMemcpy(temp, dev_xN, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) xNorm += temp[i];

  euclideanNorm<<<1, N>>>(dev_y, dev_yN);
  cudaMemcpy(temp, dev_yN, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) yNorm += temp[i];

  printf(
      "Distance: %.2lf\n"
      "x Norm: %.2lf\n"
      "y Norm: %.2lf\n",
      sqrt(dist), sqrt(xNorm), sqrt(yNorm));

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_d);
  cudaFree(dev_xN);
  cudaFree(dev_yN);

  return 0;
}
