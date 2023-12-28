target = omp_gpu_scop.out

phony: all

ppcg:
	~/ppcg/ppcg omp_gpu_scop.c

$(target):
	nvcc -Xcompiler -fopenmp -lgomp -lpng omp_gpu_scop*.cu -o omp_gpu_scop.out

all: ppcg $(target)

run: $(target)
	time ./omp_gpu_scop.out ./rem.png

clean:
	rm *.out output_cuda.png *.cu *.hu
