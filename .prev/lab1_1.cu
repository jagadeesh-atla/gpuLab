/* 21JE0192 - A JAGADEESH */
#include <stdio.h>
__global__ void helloFromGPU()
{
        printf("Hello Shubhranshu Animesh from GPU!\n");
}

int main(int argc, char **argv){


        for(int i=0; i<10; i++){
                printf("Hello Shubhranshu Animesh from CPU!\n");
        }

        helloFromGPU<<<1, 10>>>();
        cudaDeviceReset();
        return 0;
}
