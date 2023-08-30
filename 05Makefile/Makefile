NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: distanceMain.exe

distanceMain.exe: distanceMain.o DistKernel.o
	$(NVCC) $^ -o $@

distanceMain.o: distanceMain.cpp DistKernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

DistKernel.o: DistKernel.cu DistKernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
