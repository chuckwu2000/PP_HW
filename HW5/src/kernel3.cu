#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define tile_size 10

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int width, int maxIteration, int* result, size_t pitch) {
	// To avoid error caused by the floating number, use the following pseudo code
	//
	// float x = lowerX + thisX * stepX;
	// float y = lowerY + thisY * stepY;
	int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * tile_size;
	int thisY = blockIdx.y * blockDim.y + threadIdx.y;

	float c_im = lowerY + thisY * stepY;

	for(int t = 0; t < tile_size; t++)
	{
		float c_re = lowerX + (thisX + t) * stepX;
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

		//use cudaMallocPitch will add pad to make global memory access coalesced
		//pitch means (width+pad) count
		*((int*)((char*)result + thisY * pitch ) + thisX + t) = i;
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	//resX : width(1600) , resY : height(1200)
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

	int* host_img;
	cudaHostAlloc((void **) &host_img, resX * resY * sizeof(int), cudaHostAllocDefault);

	int* dev_img;
	size_t pitch;
	cudaMallocPitch((void **) &dev_img, &pitch, resX * sizeof(int), resY);

	{
		dim3 dimBlock(32, 5);
		dim3 dimGrid(resX / (32 * 10), resY / 5);
		mandelKernel<<<dimGrid, dimBlock>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, dev_img, pitch);
	}

	cudaMemcpy2D(host_img, resX * sizeof(int), dev_img, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
	cudaFree(dev_img);

	memcpy(img, host_img, resX * resY * sizeof(int));
	cudaFreeHost(host_img);
	return;
}
