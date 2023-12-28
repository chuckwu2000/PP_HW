target = gpu_scop.out

phony: all

ppcg:
	~/ppcg/ppcg gpu_scop.c

$(target):
	nvcc gpu_scop*.cu -lpng -o scop.out

all: ppcg $(target)

run: $(target)
	time ./gpu_scop.out ./rem.png

clean:
	rm gpu_scop.out output_cuda.png *.cu *.hu
