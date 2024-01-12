SHELL := /bin/bash

target1 = omp4.out
target2 = omp8.out
target3 = omp16.out

phony: all

build_target1:
	gcc -DTHREAD_4 omp.c -lpng -fopenmp -o $(target1)

build_target2:
	gcc -DTHREAD_8 omp.c -lpng -fopenmp -o $(target2)

build_target3:
	gcc -DTHREAD_16 omp.c -lpng -fopenmp -o $(target3)

all: run

run: build_target1 build_target2 build_target3
	time ./omp4.out ./rem.png
	time ./omp8.out ./rem.png
	time ./omp16.out ./rem.png

clean: 
	rm *.out
	rm output_*
