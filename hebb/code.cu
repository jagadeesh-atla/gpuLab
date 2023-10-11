#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

#define N 4

int main() {
	int x1[N] = {1, 1, -1, -1};
	int x2[N] = {-1, 1, -1, 1}; 
	int t[N]  = {-1, 1, -1, -1};
	int w1[N], w2[N], b[N];
	int yin[N], yout[N];

	for (int i = 0; i < N; ++i) {
		if (i == 0) {
			w1[i] = 0 + t[i] * x1[i];
			w2[i] = 0 + t[i] * x2[i];
			b[i]  = 0 + t[i];
		} else {
			w1[i] = w1[i - 1] + t[i] * x1[i];
			w2[i] = w2[i - 1] + t[i] * x2[i];
			b[i]  =  b[i - 1] + t[i];
		}
		yin[i] = w1[i] * x1[i] + w2[i] * x2[i] + b[i];
		yout[i] = (yin[i] > 0) ? 1 : -1;
	}
	
	printf("After epoch 1: \n");
	printf("x1\tx2\t|\tt\t|\tyin\tyout\t|\tw1\tw2\tb\n");
	for (int i = 0; i < N; ++i) {
		printf("%d\t%d\t|\t%d\t|\t%d\t%d\t|\t%d\t%d\t%d\n", x1[i], x2[i], t[i], yin[i], yout[i], w1[i], w2[i], b[i]);		
	}

	printf("\n\nVerificaion:\n");
	int v[N];
	for (int i = 0; i < N; ++i) {
		v[i] = w1[N - 1] * x1[i] + w2[N - 1] * x2[i] + b[N - 1];
		v[i] = (v[i] > 0) ? 1 : -1;
	}

	for (int i = 0; i < N; ++i) {
		printf("%d ", v[i]);
		assert(v[i] == t[i]);
	}
	printf("\n");

	return 0;
}

