import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

a = numpy.random.randn(5, 5)
a = a.astype("float32")

a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
        __global__ void doubleMatrix(float* a) {
                   int tid = threadIdx.x + threadIdx.y * 4;
                   a[tid] *= 2;
        }
    """)

func = mod.get_function("doubleMatrix")
func(a_gpu, block=(5, 5, 1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

print("Original Matrix")
print(a)
print("Doubled matrix")
print(a_doubled)
