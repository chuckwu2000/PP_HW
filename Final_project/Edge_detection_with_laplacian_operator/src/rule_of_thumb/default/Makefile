all: 
	ppcg *.c --sizes="{kernel[i]->block[16, 16]}"
	nvcc scop_host.cu scop_kernel.cu -o ppcg_lap -lpng
clean:
	rm -f ppcg_lap output* *.cu *.hu
cuda: all
	./ppcg_lap ../rem.png