/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define N 1024

// cuda code to calculate standard deviation of an array


__global__ void stdGPU(double *a, double *b, double mean) {
    int i=blockIdx.x;
    if (i<N) {
        b[i] = (a[i]-mean)*(a[i]-mean);
    }
}

int main(int argc, char **argv) {
    double a[N], b[N];
    double mu;
    mu = 0;
    double std;

    double *dev_a, *dev_b;
    cudaMalloc((void**) &dev_a, N*sizeof(double));
    cudaMalloc((void**) &dev_b, N*sizeof(double));

    for (int i=1; i<=N; i++){
        a[i-1] = 2*i - 1;
        mu += a[i-1];
    }
    mu = mu/N;

    cudaMemcpy(dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    stdGPU<<<N,1>>>(dev_a, dev_b, mu);

    cudaMemcpy(b, dev_b, N*sizeof(double), cudaMemcpyDeviceToHost);
    std = 0;
    for (int i=0; i<N; i++){
        std += b[i];
    }
    std = sqrt(std/N);
    printf("Standard Deviation: %f\n", std);

    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}