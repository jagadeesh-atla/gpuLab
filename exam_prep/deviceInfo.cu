#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0)
    printf("There are no devices.\n");
  else
    printf("Detected %d devices.\n", deviceCount);

  int dev = 0, driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device %d: \"%s\"\n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("%d.%d\n", driverVersion, runtimeVersion);

  return 0;
}
