all:
	gcc seq.c -o seq.out -lpng
	gcc omp.c -lpng -fopenmp -o omp.out
	gcc scop.c -lpng -o scop.out

seq:
	./seq.out rem.png

omp:
	./omp.out rem.png

scop:
	./scop.out rem.png

clean: 
	rm *.out
	rm output_*
