SHELL := /bin/bash

target1 = omp4_gpu_scop.out
target2 = omp8_gpu_scop.out

phony: all

$(target1):
	nvcc -Xcompiler -fopenmp -lgomp -lpng -DTHREAD_4 omp_gpu_scop*.cu -o omp4_gpu_scop.out

$(target2):
	nvcc -Xcompiler -fopenmp -lgomp -lpng -DTHREAD_8 omp_gpu_scop*.cu -o omp8_gpu_scop.out

all: run

run: $(target1) $(target2)
	time ./omp4_gpu_scop.out ./rem.png
	time ./omp8_gpu_scop.out ./rem.png

clean:
	rm *.out output_omp_cuda.png
