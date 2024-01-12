#include <stdlib.h>
#include <stdio.h> 
#include "kernel.hu"
#define ORDER 1
#define MAX_BRIGHTNESS 255
#define GetValue(a) ((a)&0xff)
#define ITERATION_NUM 10

__device__ void cuda_image_set_pixel( int x, int y, unsigned clr, int width, int height, unsigned *data ) {
	if ( x < 0 || x >= width || y < 0 || y >= height ) return;
	data[y*width+x] = clr;
}

__device__ double cuda_image_get_pixeld( int x, int y, int width, int height, unsigned *data) {
	if ( x < 0 || x >= width || y < 0 || y >= height ) return 0.;
	return (double) GetValue(data[y*width+x]);
}

__global__ void stencilCUDA(int width, int height, unsigned *d_in, unsigned *d_out, unsigned *d_buffer){
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("x: %d, y: %d\n", x, y);

    for (int i = 0; i < ITER_Y; i++ ){
        for (int j = 0; j < ITER_X; j++ ){
            if ( x+(j*(width/ITER_X)) < 1 || x+(j*(width/ITER_X)) >= width - 2 || y+(i*(height/ITER_Y)) < 1 || y+(i*(height/ITER_Y)) >= height - 2 ) 
                return;
            double val = abs(
                    8 * cuda_image_get_pixeld(x+(j*(width/ITER_X)), y+(i*(height/ITER_Y)), width, height, d_in)  -(
                    cuda_image_get_pixeld(x-1+(j*(width/ITER_X)), y-1 +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x  +(j*(width/ITER_X)), y-1 +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x+1+(j*(width/ITER_X)), y-1 +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x-1+(j*(width/ITER_X)), y   +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x+1+(j*(width/ITER_X)), y   +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x-1+(j*(width/ITER_X)), y+1 +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x  +(j*(width/ITER_X)), y+1 +(i*(height/ITER_Y)), width, height, d_in) +
                    cuda_image_get_pixeld(x+1+(j*(width/ITER_X)), y+1 +(i*(height/ITER_Y)), width, height, d_in)));
            d_buffer[(y+(i*(height/ITER_Y)))*width+x+(j*(width/ITER_X))] = val;
        }
    }

	
    // buffer = val;
    // if ( val > max ) max = val;
    // if ( val < min ) min = val;
    // val = MAX_BRIGHTNESS * (buffer - min) / (max-min);
    // cuda_image_set_pixel(x, y, val , width, height, d_out);
}

__global__ void produceOutputCUDA(int width, int height, int min, int max, unsigned *d_out, unsigned *d_buf){
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    double val = MAX_BRIGHTNESS * (cuda_image_get_pixeld(x, y, width, height, d_buf) - min) / (max-min);
    if (val > 15)
        val = 255;
    cuda_image_set_pixel(x, y, val , width, height, d_out);
}