all: 
	nvcc scop_host.cu scop_kernel.cu -o ppcg_lap -lpng
clean:
	rm -f ppcg_lap output*
cuda: all
	./ppcg_lap ../rem.png