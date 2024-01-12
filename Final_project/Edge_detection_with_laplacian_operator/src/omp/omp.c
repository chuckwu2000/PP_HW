#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <omp.h>
#include "header.h"

#define THREAD_NUMBER 1

#ifdef THREAD_4
    #undef THREAD_NUMBER
    #define THREAD_NUMBER 4
#endif

#ifdef THREAD_8
    #undef THREAD_NUMBER
    #define THREAD_NUMBER 8
#endif

#ifdef THREAD_16
    #undef THREAD_NUMBER
    #define THREAD_NUMBER 16
#endif

int main( int argc, char* argv[] )
{
	omp_set_num_threads(THREAD_NUMBER);
	Image* img1 = image_loadOMP( argv[1] );				//Input Image
	Image* img3 = image_create( img1->width, img1->height );	//Output Image
	printf("++++++++++++++++++++++++++++++\n\tOpenMP Version\t\n++++++++++++++++++++++++++++++\n");
	printf("time spent in OpenMp code in %d iterations using %d threads\n", ITERATION_NUM, THREAD_NUMBER);
	double avg_time=0;
	int i;

	for ( i=1;i<=ITERATION_NUM;i++)
	{
		stencilCodeOMP( img1, img3 );
	}
	
	image_save( img3, "output_OMP.png" );

	printf("++++++++++++++++++++++++++++++++++++\n\tEnd of OpenMP Version\t\n++++++++++++++++++++++++++++++++++++\n\n\n");
	return 0;
}
