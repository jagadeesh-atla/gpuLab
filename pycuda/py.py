import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

a = np.random.randn(16)
a = a.astype('float64')

print(a)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

module = SourceModule("""
        __global__ void double_array(double *a) {
                      int idx = blockIdx.x * blockDim.x + threadIdx.x;
                      a[idx] *= 2;
        }
""")

function = module.get_function("double_array")
function(a_gpu, block=(5, 5, 1), grid=(1, 1, 1))

a_out = np.empty_like(a)

cuda.memcpy_dtoh(a_out, a_gpu)

print(a_out)
