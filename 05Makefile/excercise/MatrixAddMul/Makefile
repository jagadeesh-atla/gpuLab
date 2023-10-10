NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: addSquare.exe

addSquare.exe: addSquare.o addSquareKernel.o
	$(NVCC) $^ -o $@

addSquare.o: addSquare.cpp operations.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

addSquareKernel.o: addSquareKernel.cu operations.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
