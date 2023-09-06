/* 21JE0192 - A JAGADEESH */
#include <cuda_runtime.h>
#include <stdio.h>
#define N 4
void displayMatrix(float *A, int nx, int ny){
int idx;
for (int i=0;i<nx; i++){
for (int j=0;j<ny; j++){
idx = i*ny+j;
printf("%f", A[idx]);
}
printf("\n");
}
return;
}

void initialData(float *ip, const int size){
int i;
for(i=0;i<size;i++){
ip[i]=i;
}
}

__device__ void transpose(float *mat_A, float *mat_B){
int col = threadIdx.x+blockIdx.x*blockDim.x;
int row = threadIdx.y+blockIdx.y*blockDim.y;
int idx_in = row*N + col;
int idx_out = row*N+col;
for (int i=0;i<blockDim.x;i+=blockDim.x){
float temp = mat_A[idx_in+i*N];
mat_B[idx_out+i]=temp;
}
}

__global__ void transposeKernel(float *mat_A, float *mat_B){
transpose(mat_A, mat_B);
}

__global__ void squareMat(float *mat){
int col = threadIdx.x+blockIdx.x*blockDim.x;
int row = threadIdx.y+blockIdx.y*blockDim.y;
if ((row<N) && (col<N)){
float Pvalue = 0;
for (int k=0;k<N;k++){
Pvalue += mat[row*N+col]*mat[row*N+col];
}
mat[row*N+col] = Pvalue;
}
}



int main(int argc, char **argv){
int nx=N;
int ny=N;
int nxy = nx*ny;
int nBytes = nxy*sizeof(float);
float *h_A, *h_B;

h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);

initialData(h_A, nxy);
initialData(h_B,  nxy);

float *mat_A, *mat_B;
cudaMalloc((void **)&mat_A, nBytes);
cudaMalloc((void **)&mat_B, nBytes);

cudaMemcpy(mat_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(mat_B, h_B, nBytes, cudaMemcpyHostToDevice);

int bdimx = 16;
int bdimy = 16;	
dim3 block(bdimx, bdimy);
dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1/block.y,1));
transposeKernel<<<grid, block>>>(mat_A, mat_B);
cudaDeviceSynchronize();

printf("Matrix A:\n");
displayMatrix(h_A, nx, ny);
printf("Matrix At:\n");
displayMatrix(h_B, nx, ny);
printf("Now multiplyig A and At\n");
squareMat<<<grid, block>>>(mat_B);
printf("Matrix after multiplication is:\n");
cudaMemcpy(h_B, mat_B,  nBytes, cudaMemcpyDeviceToHost);
displayMatrix(h_B, nx, ny);
cudaFree(mat_A);
cudaFree(mat_B);

free(h_A);
free(h_B);

cudaDeviceReset();
return(0);
}











