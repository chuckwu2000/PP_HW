#include <assert.h>
#include <stdio.h>
#include "gpu_scop_kernel.hu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#define MAX_BRIGHTNESS 255
#define ITERATION_NUM 100

int height, width;
int image_1[6000][6000] = {0};
int image_2[6000][6000] = {0};

// load a new image from png file.
int image_load( const char *file_name ) {
    png_structp png_ptr;
    png_infop info_ptr;
    FILE *fp;

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

    for (int y = 0; y < 6000; y++ ) {
        for(int x = 0; x < 6000; x++ ) {
			int c = 0;
			char* ch = (char*)&c;
			char* array = (char*)row_pointers[y];

			ch[0] = array[x];
			image_1[y][x] = c;
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
    fclose(fp);
    return 0;
}

// save png file.
int image_save( int image[6000][6000], const char* filename ) {
    png_structp png_ptr = 0;
    png_infop png_info = 0;
    png_bytep row_ptr = 0;
    int error = 1;

    FILE* file = fopen(filename, "wb");
    if ( file == 0 ) {
        return 0;
    }

    row_ptr = (png_bytep)malloc(6000);
    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, (png_voidp)NULL, NULL, NULL);

    if ( png_ptr == 0 ) {
        goto cleanup;
    }

    png_info = png_create_info_struct(png_ptr);
    if ( png_info == 0 ) {
        goto cleanup;
    }

	  png_init_io( png_ptr, file );

    png_set_IHDR(png_ptr, png_info, 6000, 6000,
        8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, png_info);

    for (int y = 0; y < 6000; y++ ) {
        for(int x = 0; x < 6000; x++ ) {
            int clr = image[y][x];
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

//StencilCode function to be applied on input image and set values to output image
void stencilCodeCUDA( int in[6000][6000], int out[6000][6000] ) {
    double* buffer = (double*)calloc(6000 * 6000,sizeof(double));
    double min = 1.0, max = 0.0;

    // for each row, column, calculating the new value using Stencil Matrix (laplacian)
	  //for(int i = 0; i < ITERATION_NUM; i++)
	  {
    	{
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

    	  double *dev_buffer;
    	  int *dev_in;
    	  
    	  cudaCheckReturn(cudaMalloc((void **) &dev_buffer, (35987998) * sizeof(double)));
    	  cudaCheckReturn(cudaMalloc((void **) &dev_in, (5999) * (6000) * sizeof(int)));
    	  
    	  cudaCheckReturn(cudaMemcpy(dev_in, in, (5999) * (6000) * sizeof(int), cudaMemcpyHostToDevice));
    	  {
    	    dim3 k0_dimBlock(16, 32);
    	    dim3 k0_dimGrid(188, 188);
    	    kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_buffer, dev_in);
    	    cudaCheckKernel();
    	  }
    	  
    	  cudaCheckReturn(cudaMemcpy(buffer, dev_buffer, (35987998) * sizeof(double), cudaMemcpyDeviceToHost));
    	  cudaCheckReturn(cudaFree(dev_buffer));
    	  cudaCheckReturn(cudaFree(dev_in));
    	}
	  }

    for (int y = 1; y < 6000 - 2; y++)
	  {
      for(int x = 1; x < 6000 - 2; x++)
		  {
			  if ( buffer[y * 6000 + x] > max ) max = buffer[y * 6000 + x];
			  if ( buffer[y * 6000 + x] < min ) min = buffer[y * 6000 + x];
      }
    }

    //Normailzing the values and set pixel value in the proper location
    for (int y = 0; y < 6000; y++ ) 
	  {
      for(int x = 0; x < 6000; x++ ) 
		  {
        double val = MAX_BRIGHTNESS * (buffer[y * 6000 + x] - min) / (max-min);
        if(val > 15)
        {
          val = 255;
        }
			  out[y][x] = val;
      }
    }
    free( buffer );
}

int main( int argc, char* argv[] )
{
	image_load( argv[1] );				//Input Image
	printf("++++++++++++++++++++++++++++++\n\tCUDA Version\t\n++++++++++++++++++++++++++++++\n");
	printf("time spend in CUDA code in %d iterations\n", ITERATION_NUM);
	int i;

	for ( i=1;i<=ITERATION_NUM;i++)
	{
		stencilCodeCUDA(image_1, image_2);
	}

	image_save( image_2, "output_cuda.png" );

	printf("++++++++++++++++++++++++++++++++++++\n\tEnd of CUDA Version\t\n++++++++++++++++++++++++++++++++++++\n\n\n");
	return 0;
}
