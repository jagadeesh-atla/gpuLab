#include "cuda_runtime.h"
#include "lab9.h"
#include "stdio.h"
#include "stdlib.h"

int main() {
  const size_t chunk = 1024 * 20;
  const size_t full_size = chunk * 100;

  doubleStream(chunk, full_size);
}
