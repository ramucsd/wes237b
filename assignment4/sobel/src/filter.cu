#include "filter.h"
#include "timer.h"

#include <iostream>

using namespace std;

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

int sobel_kernel_x[3][3] = {
	{ 1,  0, -1},
	{ 2,  0, -2},
	{ 1,  0, -1}};

__constant__ int sobel_kernel_cuda[3][3] = {
	{ 1,  0, -1},
	{ 2,  0, -2},
	{ 1,  0, -1}};

// =================== CPU Functions ===================
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	for(int i = 1; i < (height-1); i++)
	{
		for(int j = 1; j < (width-1); j++)
		{
			int16_t xval = 0;
			int16_t yval = 0;

			xval += input[((i-1)*width + j - 1)]*sobel_kernel_x[0][0];
			yval += input[((i-1)*width + j - 1)]*sobel_kernel_x[0][0];
			//xval += src_ptr[((i-1)*n_cols + j)]*sobel_kernel_x[0][1];
			yval += input[((i-1)*width + j)]*sobel_kernel_x[1][0];
			xval += input[((i-1)*width + j + 1)]*sobel_kernel_x[0][2];
			yval += input[((i-1)*width + j + 1)]*sobel_kernel_x[2][0];
			xval += input[((i)*width + j - 1)]*sobel_kernel_x[1][0];
			//yval += src_ptr[((i)*n_cols + j - 1)]*sobel_kernel_x[0][1];
			xval += input[((i)*width + j + 1)]*sobel_kernel_x[1][2];
			//yval += src_ptr[((i)*n_cols + j + 1)]*sobel_kernel_x[2][1];
			xval += input[((i+1)*width + j - 1)]*sobel_kernel_x[2][0];
			yval += input[((i+1)*width + j - 1)]*sobel_kernel_x[0][2];
			//xval += src_ptr[((i+1)*n_cols + j)]*sobel_kernel_x[2][1];
			yval += input[((i+1)*width + j)]*sobel_kernel_x[1][2];
			xval += input[((i+1)*width + j + 1)]*sobel_kernel_x[2][2];
			yval += input[((i+1)*width + j + 1)]*sobel_kernel_x[2][2];

			//cap at 255 (uint8 max val)
			uint32_t newval = (uint32_t) round(sqrt((xval*xval)+(yval*yval)));
			if(newval > 255)
			{
				output[i*width + j] = 255;
			}
			else
			{
				output[i*width + j] = (uint8_t)newval;
			}
		}
	}
}

// =================== GPU Kernel Functions ===================
__global__ void sobel_filter_kernel(const uchar *in, uchar *out, const uint height, const uint width){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = blockIdx.y;

	if((y >= height) || (x >= width))
	{
		return; //index out of bounds
	}

	int16_t xval = 0;
	int16_t yval = 0;

        for(int m = 0; m < 3; m++)
        {
		if(!(((y == 0) && (m == 0)) || ((y == (height-1)) && (m == 2) ))) //skip the top row and last row to avoid out of bounds indexing
		{
			for(int n = 0; n < 3; n++)
			{
				if(!(((x == 0) && (n == 0)) || ((x == (width-1)) && (n == 2) )))
				{
					xval += in[((y+m-1)*width + x + n - 1)]*sobel_kernel_cuda[m][n];
					yval += in[((y+m-1)*width + x + n - 1)]*sobel_kernel_cuda[n][m];
				}			
			}
		}
        }

	//cap at 255 (uchar max val)
	uint32_t newval = (uint32_t) round(sqrt((float)((xval*xval)+(yval*yval))));
	if(newval > 255)
	{
		out[y*width + x] = 255;
	}
	else
	{
		out[y*width + x] = (uchar) newval;
	}


	
}


// =================== GPU Host Functions ===================
void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	uint t = (width > 1024) ? 1024 : width;
	dim3 b(divup(width,t),height);

	sobel_filter_kernel<<<b,t>>>(input,output,height,width);
	cudaDeviceSynchronize();
}
