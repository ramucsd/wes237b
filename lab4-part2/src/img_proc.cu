#include "img_proc.h"

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================

void img_rgb2gray_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    //TODO: Convert a 3 channel RGB image to grayscale
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			uint sum = 0;
			for(int c = 0; c < channels; c++)
			{
				sum += in[i*width*channels+channels*j + c];
			}
			out[i*width + j] = sum/channels;
			
		}
	}

}

void img_invert_cpu(uchar* out, const uchar* in, const uint width, const uint height)
{
    //TODO: Invert a 8bit image
	for(int i = 0; i < width*height; i++)
	{
			out[i] = 255-in[i];
	}
}

void img_blur_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int blur_size)
{
    //TODO: Average out blur_size pixels
	int edge_size = blur_size/2;
	for(int i = edge_size; i < height-edge_size; i++)
	{
		for(int j = edge_size; j < width-edge_size; j++)
		{
			uint sum = 0;
			for(int m = -edge_size; m < edge_size; m++)
			{
				for(int n = -edge_size; n < edge_size; n++)
				{
					sum += in[(i+m)*width + j+n];
				}
			}
			
			out[i*width + j] = sum/(blur_size*blur_size);
			
		}
	}
}

// =================== GPU Kernel Functions ===================
/*
TODO: Write GPU kernel functions for the above functions
   */

__global__ void img_rgb2gray_kernel(uchar* out, const uchar* in, const int channels)
{
	//uint idx = blockIdx.y*blockDim.y*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	uint sum = 0;
	for(int c = 0; c < channels; c++)
	{
		sum += in[idx*channels+ c];
	}
	out[idx] = sum/channels;
}

__global__ void img_invert_kernel(uchar* out, const uchar* in)
{
	//uint idx = blockIdx.y*blockDim.y*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	uint idx = blockIdx.x*blockDim.x + threadIdx.x;
	out[idx] = 255-in[idx];
}

__global__ void img_blur_kernel(uchar* out, const uchar* in, const int blur_size)
{
	int yidx = blockIdx.x;
	int xidx = threadIdx.x;
	
	int width = blockDim.x;
	int height = gridDim.x;

	int edge_size = blur_size/2;

	uint sum = 0;
	uint skippedPixels = 0;
	for(int m = -edge_size; m < edge_size; m++)
	{
		for(int n = -edge_size; n < edge_size; n++)
		{
			if((yidx+m < 0) || (yidx+m >= height) || (xidx+n < 0) || (xidx+n >= width))
			{
				skippedPixels++;
			}
			else
			{
				sum += in[(yidx+m)*width + xidx+n];
			}
		}
	}
			
	out[yidx*width + xidx] = sum/((blur_size*blur_size)-skippedPixels);
			

}

// =================== GPU Host Functions ===================
/* 
TODO: Write GPU host functions that launch the kernel functions above
   */

void img_rgb2gray(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
	//int t = (width > 1024) ? 1024 : width;
	//dim3 b(divup(width,t),height);
	img_rgb2gray_kernel<<<height,width>>>(out, in, channels);
	cudaDeviceSynchronize();

}

void img_invert(uchar* out, const uchar* in, const uint width, const uint height)
{
	//int t = (width > 1024) ? 1024 : width;
	//dim3 b(divup(width,t),height);
	img_invert_kernel<<<height,width>>>(out, in);
	cudaDeviceSynchronize();
}

void img_blur(uchar* out, const uchar* in, const uint width, const uint height, const int blur_size)
{
	//int t = (width > 1024) ? 1024 : width;
	//dim3 b(divup(width,t),height);
	img_blur_kernel<<<height,width>>>(out, in,blur_size);
	cudaDeviceSynchronize();
}
