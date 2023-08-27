#ifndef MATRIXMULT_H_
#define MATRIXMULT_H_

#define TILE_WIDTH 2
__global__ void TiledMatrixMulKernel(float* MatA, float* MatB, float* MatC, int width);

__global__ void MatrixMulKernel(float* MatA, float* MatB, float* MatC, int width);

#endif
