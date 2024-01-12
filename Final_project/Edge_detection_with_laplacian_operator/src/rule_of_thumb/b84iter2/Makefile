all: 
	nvcc host.cu kernel.cu -o cuda_by_hand -lpng -g -lineinfo
clean: 
	rm cuda_by_hand output*
cuda: all
	./cuda_by_hand ../../rem.png
