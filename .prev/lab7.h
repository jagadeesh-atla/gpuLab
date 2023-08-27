#ifndef SUM_REDUCTION_H
#define SUM_REDUCTION_H

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

struct Result {
  float ans;
  float time;
};

struct Result basic_sum_reduce(float *arr, int n);
struct Result improved_sum_reduce(float *arr, int n);

#endif
