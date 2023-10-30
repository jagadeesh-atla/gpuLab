import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

N = 1 << 4

a = np.random.randn(N)
b = np.random.randn(N)

a = a.astype('float64')
b = b.astype('float64')

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

module = SourceModule("""
              __global__ void vecAddGPU(double *a, double *b, double *c, int N) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid < N) c[tid] = a[tid] + b[tid];
              }
            """)

c = np.empty_like(a)
c_gpu = cuda.mem_alloc(c.nbytes)

function = module.get_function("vecAddGPU")
block_size = 128
grid_size = (N + block_size - 1) // block_size

function(a_gpu, b_gpu, c_gpu, np.int32(N), block=(
    block_size, 1, 1), grid=(grid_size, 1))

cuda.memcpy_dtoh(c, c_gpu)

print(a)
print(b)
print(c)
