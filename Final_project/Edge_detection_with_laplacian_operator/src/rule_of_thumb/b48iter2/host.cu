#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include "kernel.hu"

#define ORDER 1

#define MAX_BRIGHTNESS 255
#define GetValue(a) ((a)&0xff)
#define ITERATION_NUM 100

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

Image* image_create( unsigned width, unsigned height ) {
	unsigned size = width * height * sizeof(unsigned);
	Image* image = (Image*)malloc(sizeof(unsigned)+size);
	memset( image->data, 0, size );
	image->width = width;
	image->height = height;
	return image;
}

//empty memory occupied by image
void image_destroy( Image* image ) {
	free( image );
}

// retrieve the pixel value.
unsigned image_get_pixel( Image* image, int x, int y ) {
	if ( x < 0 || x >= image->width || y < 0 || y >= image->height ) return 0;
	return image->data[y*image->width+x];
}

// retrieve a particular value as a double
double image_get_pixeld( Image* image, int x, int y ) {
	if ( x < 0 || x >= image->width || y < 0 || y >= image->height ) return 0.;
	return (double)GetValue(image->data[y*image->width+x]);
}

// set a pixel value.
void image_set_pixel( Image* image, int x, int y, unsigned clr ) {
	if ( x < 0 || x >= image->width || y < 0 || y >= image->height ) return;
	image->data[y*image->width+x] = clr;
}

// load a new image from png file.
Image* image_load( const char *file_name ) {
	png_structp png_ptr;
	png_infop info_ptr;
	png_uint_32 width, height;
	FILE *fp;
	Image* image = NULL;
	unsigned x,y;

	if ((fp = fopen(file_name, "rb")) == NULL)
		return 0;

	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (png_ptr == NULL) {
		fclose(fp);
		return 0;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fclose(fp);
		png_destroy_read_struct(&png_ptr, png_infopp_NULL, png_infopp_NULL);
		return 0;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
		fclose(fp);
		return 0;
	}

	png_init_io(png_ptr, fp);
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_ALPHA, NULL);

    int channel = png_get_channels(png_ptr, info_ptr);
    int depth = png_get_bit_depth(png_ptr, info_ptr);
    int color = png_get_color_type(png_ptr, info_ptr);
    printf("channel : %d, depth : %d, color : %d\n",channel,depth,color);
	png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	width = png_get_image_width(png_ptr, info_ptr);
	image = image_create( width, height );

	for ( y = 0; y < height; y++ ) {
		for( x = 0; x < width; x++ ) {
			unsigned c = 0;
			unsigned char* ch = (unsigned char*)&c;
			unsigned char* array = row_pointers[y];

			ch[0] = array[x];
			image_set_pixel(image, x, y, c);
		}
	}

	
	   
	png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
	fclose(fp);
	return image;
}

// save png file.
int image_save( Image* image, const char* filename ) {
	png_structp png_ptr = 0;
	png_infop png_info = 0;
	png_bytep row_ptr = 0;
	int x,y;
	int error = 1;

	FILE* file = fopen(filename, "wb");
	if ( file == 0 ) {
		return 0;
	}

	row_ptr = (png_bytep)malloc(image->width);
	png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, (png_voidp)NULL, NULL, NULL);

	if ( png_ptr == 0 ) {
		goto cleanup;
	}

	png_info = png_create_info_struct(png_ptr);
	if ( png_info == 0 ) {
		goto cleanup;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		goto cleanup;
	}

	png_init_io( png_ptr, file );

	png_set_IHDR(png_ptr, png_info, image->width, image->height, 
		8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png_ptr, png_info);

	for ( y = 0; y < image->height; y++ ) {
		for( x = 0; x < image->width; x++ ) {
			unsigned clr = image_get_pixel( image, x, y );
			row_ptr[x] = ( clr );
		}
		png_write_row(png_ptr, row_ptr);
	}
	png_write_end(png_ptr, png_info);
	error = 0;
	cleanup:
	fclose(file);
	if ( png_ptr ) { 
		png_destroy_write_struct(&png_ptr, &png_info);
	}    
	free( row_ptr );
	return !error;
}

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

int main( int argc, char* argv[] )
{
	//clock_t begin = clock();
	Image* img1 = image_load( argv[1] );				//Input Image
	Image* img3 = image_create( img1->width, img1->height );	//Output Image
	printf("++++++++++++++++++++++++++++++++++\n\tAccelerated Version\t\n++++++++++++++++++++++++++++++++++\n");	
	printf("time spent in sequential code with %d iterations\n", ITERATION_NUM);
	int i;
	unsigned *d_in;
	unsigned *d_out;
	unsigned *d_buf;
	float gpu_time = 0.0f;
	float avg_time = 0.0f;
	cudaEvent_t start, stop;
	cudaCheckReturn(cudaEventCreate(&start));
	cudaCheckReturn(cudaEventCreate(&stop));
	dim3 dimBlock( BLOCK_X, BLOCK_Y );
	dim3 dimGrid1( (img1->width) / dimBlock.x / ITER_X, (img1->height) / dimBlock.y / ITER_Y );
	dim3 dimGrid2( (img1->width) / dimBlock.x , (img1->height) / dimBlock.y  );
	unsigned *h_buf = (unsigned *)malloc(img1->height * img1->width * sizeof( unsigned ) );

	cudaCheckReturn(cudaMalloc( (void**)&d_in, img1->height * img1->width * sizeof( unsigned ) ));
	cudaCheckReturn(cudaMalloc( (void**)&d_out, img1->height * img1->width * sizeof( unsigned ) ));
	cudaCheckReturn(cudaMalloc( (void**)&d_buf, img1->height * img1->width * sizeof( unsigned ) ));
	cudaCheckReturn(cudaMemcpy( d_in, img1->data, img1->height * img1->width * sizeof( unsigned ), cudaMemcpyHostToDevice ));
	//repeating process to extract best time
	for ( i = 1; i<=ITERATION_NUM; i++)
	{
		cudaEventRecord(start, 0);

		stencilCUDA<<<dimGrid1, dimBlock>>>( img1->width, img1->height, d_in, d_out, d_buf );
		cudaCheckKernel();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaCheckReturn(cudaEventElapsedTime(&gpu_time, start, stop));
		// printf("iteration: %d \t time: %f msec\n",i, gpu_time);
		avg_time += gpu_time;

		/*
		 * Post, non-stencil part
		 */	
		cudaMemcpy( h_buf, d_buf, img1->height * img1->width * sizeof( unsigned ), cudaMemcpyDeviceToHost );
		auto result = std::minmax_element(h_buf, h_buf + img1->height * img1->width);
		// printf("min: %d, max: %d\n", *result.first, *result.second);
		
		produceOutputCUDA<<<dimGrid2, dimBlock>>>(img1->width, img1->height, *result.first,  *result.second, d_out, d_buf);
		cudaCheckKernel();

	}
	cudaCheckReturn(cudaMemcpy( img3->data, d_out, img1->height * img1->width * sizeof( unsigned ), cudaMemcpyDeviceToHost ));	
	printf("++++++++++++++++++++++++++++++++++++\nExecute %d timing is: %f sec\n++++++++++++++++++++++++++++++++++++\n",ITERATION_NUM, avg_time/CLOCKS_PER_SEC);
	image_save( img3, "output_cuda_scratch.png" );

	//clock_t end1 = clock();
	//double time_spent1 = (double)(end1 - avg_time) / CLOCKS_PER_SEC;
	//printf("time spent in saving is: %f sec\n", time_spent1);
	printf("++++++++++++++++++++++++++++++++++++\n\tEnd of GPU Version\t\n++++++++++++++++++++++++++++++++++++\n\n\n");
	return 0;
}