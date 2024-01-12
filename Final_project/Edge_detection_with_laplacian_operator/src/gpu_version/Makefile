SHELL := /bin/bash

target = gpu_scop.out

phony: all

build_target:
	nvcc gpu_scop*.cu -lpng -o $(target)

all: run

run: build_target
	time ./$(target) ./rem.png

clean:
	rm gpu_scop.out output_cuda.png
