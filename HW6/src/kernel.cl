__kernel void convolution(int filterWidth, __global float *filter, int imageHeight, int imageWidth, 
                          __global float *inputImage, __global float *outputImage) 
{
	int thread_index = get_global_id(0);
	int row_pos = thread_index / imageWidth;
	int col_pos = thread_index % imageWidth;

	int halffilterSize = filterWidth / 2;
	float sum = 0;
	
	for(int i = -halffilterSize; i <= halffilterSize; i++)
	{
		for(int j = -halffilterSize; j <= halffilterSize; j++)
		{
			if(filter[(i + halffilterSize) * filterWidth + (j + halffilterSize)] != 0)
			{
				if(row_pos + i >= 0 && row_pos + i < imageHeight && col_pos + j >= 0 && col_pos + j < imageWidth)
				{
					sum += inputImage[(row_pos + i) * imageWidth + (col_pos + j)] * filter[(i + halffilterSize) * filterWidth + (j + halffilterSize)];
				}
			}
		}
	}

	outputImage[row_pos * imageWidth + col_pos] = sum;
}
