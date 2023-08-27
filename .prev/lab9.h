#ifndef SUM_REDUCTION_H
#define SUM_REDUCTION_H

#include "cuda_runtime.h"
#include "stdlib.h"
#define BD 256

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

void doubleStream(const size_t, const size_t);

#endif
