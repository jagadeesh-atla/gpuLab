#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define N 6
#define MASK_DIM 3
#define MASK_OFFSET (MASK_DIM / 2)

__constant__ int mask[MASK_DIM * MASK_DIM];

void getFiltersFromFile(const char *fileName, vector<vector<int>> &filters) {
  ifstream file(fileName);
  if (!file.is_open()) {
    cerr << "Unable to open file " << fileName << ".\n ";
    return;
  }
  vector<int> currentFilter;
  int value;
  while (file >> value) {
    currentFilter.push_back(value);

    if (file.peek() == '\n') {
      file.seekg(2, ios_base::cur);
      if (file.peek() == '\n') {
        filters.push_back(currentFilter);
        currentFilter.clear();
      } else {
        file.seekg(-2, ios_base::cur);
      }
    }
  }

  file.close();
  return;
}

__global__ void convolution_2d(int *matrix, int *result, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size && col < size) {
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    int temp = 0;

    for (int i = 0; i < MASK_DIM; i++)
      for (int j = 0; j < MASK_DIM; j++)
        if ((start_r + i) >= 0 && (start_r + i) < N)
          if ((start_c + j) >= 0 && (start_c + j) < N)
            temp += matrix[(start_r + i) * N + (start_c + j)] *
                    mask[i * MASK_DIM + j];

    result[row * N + col] = temp;
  }
}

void calculate(int *input, int *filter, int *result, int size,
               int filter_size) {
  size_t bytes_n = size * size * sizeof(int);
  size_t bytes_m = filter_size * filter_size * sizeof(int);

  int *d_input;
  int *d_result;
  cudaMalloc(&d_input, bytes_n);
  cudaMalloc(&d_result, bytes_n);

  cudaMemcpy(d_input, input, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, filter, bytes_m);

  int THREADS = 16;
  int BLOCKS = (size + THREADS - 1) / THREADS;

  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  convolution_2d<<<grid_dim, block_dim>>>(d_input, d_result, size);

  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_result);

  return;
}

void verify_result(int *m, int *mask, int *result, int size) {
  // Temp value for accumulating results
  int temp;

  // Intermediate value for more readable code
  int offset_r;
  int offset_c;

  // Go over each row
  for (int i = 0; i < size; i++) {
    // Go over each column
    for (int j = 0; j < size; j++) {
      // Reset the temp variable
      temp = 0;

      // Go over each mask row
      for (int k = 0; k < MASK_DIM; k++) {
        // Update offset value for row
        offset_r = i - MASK_OFFSET + k;

        // Go over each mask column
        for (int l = 0; l < MASK_DIM; l++) {
          // Update offset value for column
          offset_c = j - MASK_OFFSET + l;

          // Range checks if we are hanging off the matrix
          if (offset_r >= 0 && offset_r < size) {
            if (offset_c >= 0 && offset_c < size) {
              // Accumulate partial results
              temp += m[offset_r * size + offset_c] * mask[k * MASK_DIM + l];
            }
          }
        }
      }
      // Fail if the results don't match
      assert(result[i * size + j] == temp);
    }
  }
}

int main(void) {
  vector<vector<int>> filt;
  getFiltersFromFile("filters/X.txt", filt);

  //   for (auto x : filt) {
  //     for (int y : x) {
  //       cout << y << " ";
  //     }
  //     cout << endl;
  //   }
  //   return 0;

  int *input = new int[N * N];
  int *filter = new int[MASK_DIM * MASK_DIM];

  for (int i = 0; i < N * N; i++) {
    input[i] = rand() % 2;
    cout << input[i] << " ";
  }
  cout << endl;

  for (int i = 0; i < MASK_DIM * MASK_DIM; i++) {
    filter[i] = filt[0][i];
    cout << filter[i] << " ";
  }
  cout << endl;

  int *output = new int[N * N];
  calculate(input, filter, output, N, MASK_DIM);
  cudaDeviceSynchronize();

  verify_result(input, filter, output, N);

  for (int i = 0; i < MASK_DIM * MASK_DIM; ++i) {
    cout << output[i] << " ";
  }

  delete[] input;
  delete[] filter;
  delete[] output;

  return 0;
}
