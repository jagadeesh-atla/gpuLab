#include "stdio.h"
#include "stdlib.h"
#include "lab7.h"
#define N 10000

void init_data(float *arr, int n) {
  for (int i = 0; i < n; ++i) {
    arr[i] = i;
  }
}

int main() {
  struct Result ans;
  float *arr;
  int arr_size = N * sizeof(float);
  arr = (float *)malloc(arr_size);
  init_data(arr, N);

  ans = basic_sum_reduce(arr, N);
  printf("Sum: %f\n", ans.ans);
  printf("Execution Time: %f\n", ans.time);

  ans = improved_sum_reduce(arr, N);
  printf("Sum: %f\n", ans.ans);
  printf("Execution Time: %f\n", ans.time);

  free(arr);
}
