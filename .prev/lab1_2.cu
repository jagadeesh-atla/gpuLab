/* 21JE0192 - A JAGADEESH */
#include <stdio.h>
__global__ void printGPU(){
        printf("GPU Computing Lab\nExperiment 2.1\nDate: 10-08-2022\n");
}

int main(int argc, char **argv) {
        printf("======CPU======\n");
        for (int i=0; i<4; i++) {
                printf("GPU Computing Lab\nExperiment 2.1\nDate: 10-08-2022\n");
        }

        printf("======GPU======\n");
        printGPU<<<1, 4>>>();
        cudaDeviceReset();
        return 0;
}