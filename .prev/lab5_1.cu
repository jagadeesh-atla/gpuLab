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

__device__ void matAdd(float *mat_A, float *mat_B, float *mat_C){
int col = threadIdx.x+blockIdx.x*blockDim.x;
int row = threadIdx.y+blockIdx.y*blockDim.y;
int idx = row*N + col;
if (col<N && row<N){
mat_C[idx]=mat_B[idx]+mat_A[idx];
}
}

__global__ void matAddKernel(float *mat_A, float *mat_B, float *mat_C){
matAdd(mat_A, mat_B, mat_C);
}

__global__ void squareMat(float *mat){
int col = threadIdx.x+blockIdx.x*blockDim.x;
int row = threadIdx.y+blockIdx.y*blockDim.y;
if ((row<N) && (col<N)){
float Pvalue = 0;
for (int k=0;k<N;k++){
Pvalue += mat[row*N+col]*mat[k*N+col];
}
mat[row*N+col] = Pvalue;
}
}



int main(int argc, char **argv){
int nx=N;
int ny=N;
int nxy = nx*ny;
int nBytes = nxy*sizeof(float);
float *h_A, *h_B, *h_C;

h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);
h_C = (float *)malloc(nBytes);

initialData(h_A, nxy);
initialData(h_B,  nxy);

float *mat_A, *mat_B, *mat_C;
cudaMalloc((void **)&mat_A, nBytes);
cudaMalloc((void **)&mat_B, nBytes);
cudaMalloc((void **)&mat_C, nBytes);

cudaMemcpy(mat_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(mat_B, h_B, nBytes, cudaMemcpyHostToDevice);

int bdimx = 16;
int bdimy = 16;	
dim3 block(bdimx, bdimy);
dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1/block.y,1));

matAddKernel<<<grid, block>>>(mat_A, mat_B, mat_C);
cudaDeviceSynchronize();

cudaMemcpy(h_C, mat_C, nBytes, cudaMemcpyDeviceToHost);
printf("Matrix A:\n");
displayMatrix(h_A, nx, ny);
printf("Matrix B:\n");
displayMatrix(h_B, nx, ny);
printf("Addition of A and B:\n");
displayMatrix(h_C, nx, ny);
printf("Now squaring Matrix C\n");
squareMat<<<grid, block>>>(mat_C);
cudaMemcpy(h_C, mat_C, nBytes, cudaMemcpyDeviceToHost);
printf("Matrix after squaring is:\n");
displayMatrix(h_C, nx, ny);
cudaFree(mat_A);
cudaFree(mat_B);
cudaFree(mat_C);

free(h_A);
free(h_B);
free(h_C);

cudaDeviceReset();
return(0);
}











