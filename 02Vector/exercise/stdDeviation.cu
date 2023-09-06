/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define N 1024

__global__ void distance(double *a, double *mean, double *c) {
  int i = threadIdx.x;
  if (i < N) c[i] = (a[i] - *mean) * (a[i] - *mean);
}

int main() {
  double y[N], dist[N], mean = 0, dev = 0;

  for (int i = 1; i <= N; ++i) {
    y[i - 1] = 2 * i + 1;
    mean += y[i - 1];
  }

  mean /= N;

  double *dev_y, *dev_d, *dev_mean;

  cudaMalloc((void **)&dev_y, N * sizeof(double));
  cudaMalloc((void **)&dev_d, N * sizeof(double));
  cudaMalloc((void **)&dev_mean, sizeof(double));

  cudaMemcpy(dev_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mean, &mean, sizeof(double), cudaMemcpyHostToDevice);

  distance<<<1, N>>>(dev_y, dev_mean, dev_d);

  cudaMemcpy(dist, dev_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) dev += dist[i];

  dev /= N;

  dev = sqrt(dev);

  printf("Standard Deviation: %.2lf\n", dev);

  cudaFree(dev_y);
  cudaFree(dev_mean);
  cudaFree(dev_d);

  return 0;
}
