#include "funcDef.h"

void initialData(float *ip, const int size){
	int i;
 	for (i = 0; i < size; ++i) 
		ip[i] = ((float) rand() / (float) RAND_MAX);
	return;
}
