#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int maxIteration, int* result) {
	// To avoid error caused by the floating number, use the following pseudo code
	//
	// float x = lowerX + thisX * stepX;
	// float y = lowerY + thisY * stepY;
	int thisX = blockIdx.x * blockDim.x + threadIdx.x;
	int thisY = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_index = thisY * width + thisX;

	float c_re = lowerX + thisX * stepX;
	float c_im = lowerY + thisY * stepY;

	float z_re = c_re, z_im = c_im;
	int i;
	for(i = 0; i < maxIteration; ++i)
	{
		if(z_re * z_re + z_im * z_im > 4.f)
			break;

		float new_re = z_re * z_re - z_im * z_im;
		float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}

	result[thread_index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	//resX : width(1600) , resY : height(1200)
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

	cudaHostRegister(img, resX * resY * sizeof(int), cudaHostRegisterMapped);

	int* dev_img;
	//last arg : flag(must be 0 for now -> cuda_documentation said)
	cudaHostGetDevicePointer((void **) &dev_img, (void *) img, 0);

	{
		dim3 dimBlock(32, 6);
		dim3 dimGrid(resX / 32, resY / 6);
		mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, dev_img);
		//I think can remove it? synchronize looks like don't have any impact on result or performance
		cudaDeviceSynchronize();
	}

	cudaHostUnregister(img);
	return;
}
