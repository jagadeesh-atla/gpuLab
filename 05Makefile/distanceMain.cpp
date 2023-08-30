#include <stdlib.h>

#include "DistKernel.h"

#define N 16

float scale(int i, int n) { return ((float)i) / (n - 1); }

int main() {
  const float ref = 0.5f;

  float *in = (float *)calloc(N, sizeof(float));
  float *out = (float *)calloc(N, sizeof(float));

  for (int i = 0; i < N; ++i) {
    in[i] = scale(i, N);
  }

  distanceArray(out, in, ref, N);

  free(in);
  free(out);

  return 0;
}
