#include <CL/cl.h>
#include <stdio.h>

#define arraySize 1024

int main() {
  // Initialize OpenCL
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  // Data
  // const int arraySize = 1024;
  int arrayA[arraySize];
  int arrayB[arraySize];
  int result[arraySize];

  for (int i = 0; i < arraySize; i++) {
    arrayA[i] = i;
    arrayB[i] = i * 2;
  }

  // Create OpenCL buffers
  cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(int) * arraySize, NULL, NULL);
  cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(int) * arraySize, NULL, NULL);
  cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       sizeof(int) * arraySize, NULL, NULL);

  // Write data to buffers
  clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(int) * arraySize,
                       arrayA, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(int) * arraySize,
                       arrayB, 0, NULL, NULL);

  // Load and compile OpenCL program
  const char* source =
      "__kernel void addArrays(__global int* A, __global int* B, __global int* "
      "C) {"
      "    int id = get_global_id(0);"
      "    C[id] = A[id] + B[id];"
      "}";
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "addArrays", NULL);

  // Set kernel arguments
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);

  // Execute the kernel
  size_t globalWorkSize = arraySize;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL,
                         NULL);
  clFinish(queue);

  // Read the result back
  clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(int) * arraySize,
                      result, 0, NULL, NULL);

  // Clean up
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferResult);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // Print the result
  for (int i = 0; i < arraySize; i++) {
    printf("Result[%d] = %d\n", i, result[i]);
  }

  return 0;
}
