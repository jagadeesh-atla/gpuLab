from numba import guvectorize
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import numpy as np

a_gpu = gpuarray.to_gpu(np.random.randn(5, 5).astype("float32"))

a_doubled = (2 * a_gpu).get()
print("Original Matrix")
print(a_gpu)
print("Doubled Matrix by gpuarray")
print(a_doubled)


@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'], '(m,n), (n,p)->(m,p)')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


dim = 10
A = np.random.randint(dim, size=(dim, dim))
B = np.random.randint(dim, size=(dim, dim))
C = np.empty_like(A)
matmul(A, B, C)

print(A, B, C, sep='\n')
assert (C == np.dot(A, B)).all()
