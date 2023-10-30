import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

n = 1 << 2

matrix_a = np.random.randn(n, n)
matrix_b = np.random.randn(n, n)

matrix_a = matrix_a.astype(np.float32)
matrix_b = matrix_b.astype(np.float32)

matrix_a_gpu = cuda.mem_alloc(matrix_a.nbytes)
matrix_b_gpu = cuda.mem_alloc(matrix_b.nbytes)

cuda.memcpy_htod(matrix_a_gpu, matrix_a)
cuda.memcpy_htod(matrix_b_gpu, matrix_b)

mod = SourceModule("""
                   __global__ void matrix_add(float *a, float *b, float *c, int n) {
                        int i = threadIdx.x + blockIdx.x * blockDim.x;
                        int j = threadIdx.y + blockIdx.y * blockDim.y;
                        int idx = i + j * n;
                        
                        if (i < n && j < n) {
                            c[idx] = a[idx] + b[idx];
                        }
                    }
                   """)

matrix_add_kernel = mod.get_function("matrix_add")

result = np.empty_like(matrix_a)
result_gpu = cuda.mem_alloc(result.nbytes)

block_size = (16, 16, 1)
grid_size = (n // block_size[0] + 1, n // block_size[1] + 1, 1)

matrix_add_kernel(matrix_a_gpu, matrix_b_gpu, result_gpu,
                  np.int32(n), block=block_size, grid=grid_size)

cuda.memcpy_dtoh(result, result_gpu)

print(result)
