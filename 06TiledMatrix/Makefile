NVCC = nvcc

Tiled-Mat-multGPU.exe: Tiled-Mat-multGPU.o initialDataMatAB.o funcDef.h
	$(NVCC) Tiled-Mat-multGPU.cu initialDataMatAB.o -o Tiled-Mat-multGPU.exe

Tiled-Mat-multGPU.o: Tiled-Mat-multGPU.cu funcDef.h initialDataMatAB.o
	$(NVCC) -c Tiled-Mat-multGPU.cu initialDataMatAB.o

initialDataMatAB.o: funcDef.h initialDataMatAB.cu
	$(NVCC) -c initialDataMatAB.cu

clean:
	rm -f Tiled-Mat-multGPU.exe

