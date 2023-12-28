all:
	gcc omp.c -lpng -fopenmp -o omp.out

seq:
	./seq.out rem.png

omp:
	/usr/bin/time ./omp.out rem.png

scop:
	./scop.out rem.png

clean: 
	rm *.out
	rm output_*
