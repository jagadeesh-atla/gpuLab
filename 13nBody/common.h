#include <ctime>

#ifndef _COMMON_H
#define _COMMON_H

#ifndef __linux__
#include <windows.h>
#endif

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
      exit(1);                                                                 \
    }                                                                          \
  }

inline double seconds() {
#ifdef __linux__
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) +
         static_cast<double>(ts.tv_nsec) * 1e-9;
#else
  // For Windows, use QueryPerformanceCounter
  LARGE_INTEGER frequency;
  LARGE_INTEGER start;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&start);
  return static_cast<double>(start.QuadPart) /
         static_cast<double>(frequency.QuadPart);
#endif
}

#endif  // _COMMON_H
