#include <cuda_runtime.h>

#include <stdio.h>

#define SIZE (100*1024*104)

void* big_block(int size) {
	unsigned char* data = (unsigned char*) malloc(size);
	for (int i = 0; i < size; ++i)
		data[i] = rand();
	return data;	
}

int main() {
	unsigned char *str = (unsigned char *)big_block(SIZE);

	

	return 0;
}

