#include <cuda_runtime.h>

#include <stdio.h>

float kahan(float arr[], int n) {
	float sum = 0.0, c = 0.0;
	int i = 0;

	while (i < n) {
		float y = arr[i] - c;
		float t = sum + y;
		c = (t - sum) - y;
		sum = t;		
		i = i + 1;
	}	

	return sum;
}

int main() {
	float arr[] = {1.3435, -0.00012, -0.0001, 92487341, 8794391624.4234, 0.0000234};

	int n = sizeof(arr) / sizeof(float);

	float sum = kahan(arr, n);

	for (int i = 0; i < n; ++ i) {
		printf("%f ",arr[i]);
	 }
	printf("\n");

	printf("Sum =  %f\n", sum);

	return 0;
}

