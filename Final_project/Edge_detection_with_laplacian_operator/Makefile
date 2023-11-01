all:
	gcc seq.c -o seq.out -lpng
	gcc omp.c -lpng -fopenmp -o omp.out

seq:
	./seq.out rem.png

omp:
	./omp.out rem.png 

clean: 
	rm *.out
	rm output_*
