#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

	//Create command_queue
	//clCreateCommandQueue(cl_context, cl_device_id, queue_properties, errcode)
	cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);
	CHECK(status, "clCreateCommandQueue");

	//Create device memory to input image
	//clCreateBuffer(cl_context, mem_flags, size, host_ptr, errcode)
	cl_mem d_input_image = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);
	CHECK(status, "clCreateBuffer");

	//Create device memory to output image
	cl_mem d_output_image = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);
	CHECK(status, "clCreateBuffer");

	//Create device memory to filter
	cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, &status);
	CHECK(status, "clCreateBuffer");

	//Transfer inputImage to device
	//clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr, num_events, cl_event_list, cl_event)
	status = clEnqueueWriteBuffer(command_queue, d_input_image, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), inputImage, 0, NULL, NULL);
	CHECK(status, "clEnqueueWriteBuffer");

	//Transfer filter to device
	status = clEnqueueWriteBuffer(command_queue, d_filter, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
	CHECK(status, "clEnqueueWriteBuffer");

	//Create kernel
	//clCreateKernel(cl_program, kerenl_name -> "must same to kernel's name", errcode)
	cl_kernel kernel = clCreateKernel(*program, "convolution", status);
	CHECK(status, "clCreateKernel");

	//Set kernel arguments
	//clSetKernelArg(cl_kernel, arg_index, arg_size, arg_value)
	clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&filterWidth);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_filter);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&imageHeight);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&imageWidth);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&d_input_image);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&d_output_image);

	//Set global_work_size(total thread number) & local_work_size(total thread in each group)
	size_t global_size = imageHeight * imageWidth;
	size_t local_size = 64;

	//Launch kernel
	//clEnqueueNDRangeKernel(command_queue, cl_kernel, work_dim, gloabl_work_offset, global_work_size, local_work_size, event...)
	status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK(status, "clEnqueueNDRangeKernel");

	//Get output image back from device
	//clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr, num_events, cl_event_list, cl_event)
	status = clEnqueueReadBuffer(command_queue, d_output_image, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), outputImage, 0, NULL, NULL);
	CHECK(status, "clEnqueueReadBuffer");

	//Clean up & wait for all command complete
	status = clFlush(command_queue);
	CHECK(status, "clFlush");
	status = clFinish(command_queue);
	CHECK(status, "clFinish");

	//Release all OpenCL objects
	clReleaseKernel(kernel);
	clReleaseMemObject(d_input_image);
	clReleaseMemObject(d_output_image);
	clReleaseMemObject(d_filter);
	clReleaseCommandQueue(command_queue);
	return 0;
}
