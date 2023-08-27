#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define N 1024

__global__ void distanceGPU(double *a, double *b, double *c) {
    int i=blockIdx.x;
    if (i<N) {
        c[i] = (a[i]-b[i])*(a[i]-b[i]);
    }
}

__global__ void euclideanNormGPU(double *a, double *c) {
    int i=blockIdx.x;
    if (i<N) {
        c[i] = a[i]*a[i];
    }
}

int main(int argc, char **argv) {
    double a[N], b[N], c[N], d;

    double *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**) &dev_a, N*sizeof(double));
    cudaMalloc((void**) &dev_b, N*sizeof(double));
    cudaMalloc((void**) &dev_c, N*sizeof(double));

    for (int i=1; i<=N; i++){
        a[i-1] = i*i;
        b[i-1] = 2*i + 1;
    }

    cudaMemcpy(dev_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(double), cudaMemcpyHostToDevice);

    distanceGPU<<<N,1>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(double), cudaMemcpyDeviceToHost);
    d = 0;
    for (int i=0; i<N; i++) {
        d += c[i];
        // printf("%d ", c[i]);
    }

    printf("Distance = %f\n", sqrt(d));

    euclideanNormGPU<<<N,1>>>(dev_a, dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(double), cudaMemcpyDeviceToHost);
    d = 0;
    for (int i=0; i<N; i++) {
        d += c[i];
        // printf("%d ", c[i]);
    }
    printf("Norm a = %f\n", sqrt(d));

    euclideanNormGPU<<<N,1>>>(dev_b, dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(double), cudaMemcpyDeviceToHost);
    d = 0;
    for (int i=0; i<N; i++) {
        d += c[i];
        //printf("%d ", c[i]);
    }
    printf("Norm b = %f\n", sqrt(d));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
