
#include "sobel.h"
#include <stdio.h>
#include <iostream>
#include "arm_neon.h"

using namespace std;
using namespace cv;


int sobel_kernel_x[3][3] = {
	{ 1,  0, -1},
	{ 2,  0, -2},
	{ 1,  0, -1}};

/*
int sobel_kernel_y[3][3] = {
	{ 1,  2, 1},
	{ 0,  0, 0},
	{ -1, -2, -1}};
	*/

int16_t sobel_kernel_x_vec[8] = {1, 0, -1, 2, -2, 1, 0, -1};
int16_t sobel_kernel_y_vec[8] = {1, 2, 1, 0, 0, -1, -2, -1};


void sobel(const Mat& src, Mat& dst)
{
	const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = dst.ptr<uint8_t>();

	int n_cols = src.cols;
	int n_rows = src.rows;
	//uint32_t max;

	//uint32_t temp[n_rows][n_cols];
	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_cols; j++)
		{
			int16_t xval = 0;
			int16_t yval = 0;
        	for(int m = 0; m < 3; m++)
        	{
				if(!(((i == 0) && (m == 0)) || ((i == (n_rows-1)) && (m == 2) ))) //skip the top row and last row to avoid out of bounds indexing
				{
					for(int n = 0; n < 3; n++)
					{
						if(!(((j == 0) && (n == 0)) || ((j == (n_cols-1)) && (n == 2) )))
						{
							xval += src_ptr[((i+m-1)*n_cols + j + n - 1)]*sobel_kernel_x[m][n];
							yval += src_ptr[((i+m-1)*n_cols + j + n - 1)]*sobel_kernel_x[n][m];
						}
							
					}
				}

        	}
			
			//cap at 255 (uint8 max val)
			uint32_t newval = (uint32_t) round(sqrt((xval*xval)+(yval*yval)));
			if(newval > 255)
			{
				dst_ptr[i*n_cols + j] = 255;
			}
			else
			{
				dst_ptr[i*n_cols + j] = (uint8_t)newval;
			}
		}
	}
}


void sobel_unroll(const Mat& src, Mat& dst)
{
	const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = dst.ptr<uint8_t>();

	int n_cols = src.cols;
	int n_rows = src.rows;
	//uint32_t max;

	//uint32_t temp[n_rows][n_cols];
	for(int i = 1; i < (n_rows-1); i++)
	{
		for(int j = 1; j < (n_cols-1); j++)
		{
			int16_t xval = 0;
			int16_t yval = 0;

			xval += src_ptr[((i-1)*n_cols + j - 1)]*sobel_kernel_x[0][0];
			yval += src_ptr[((i-1)*n_cols + j - 1)]*sobel_kernel_x[0][0];
			//xval += src_ptr[((i-1)*n_cols + j)]*sobel_kernel_x[0][1];
			yval += src_ptr[((i-1)*n_cols + j)]*sobel_kernel_x[1][0];
			xval += src_ptr[((i-1)*n_cols + j + 1)]*sobel_kernel_x[0][2];
			yval += src_ptr[((i-1)*n_cols + j + 1)]*sobel_kernel_x[2][0];
			xval += src_ptr[((i)*n_cols + j - 1)]*sobel_kernel_x[1][0];
			//yval += src_ptr[((i)*n_cols + j - 1)]*sobel_kernel_x[0][1];
			xval += src_ptr[((i)*n_cols + j + 1)]*sobel_kernel_x[1][2];
			//yval += src_ptr[((i)*n_cols + j + 1)]*sobel_kernel_x[2][1];
			xval += src_ptr[((i+1)*n_cols + j - 1)]*sobel_kernel_x[2][0];
			yval += src_ptr[((i+1)*n_cols + j - 1)]*sobel_kernel_x[0][2];
			//xval += src_ptr[((i+1)*n_cols + j)]*sobel_kernel_x[2][1];
			yval += src_ptr[((i+1)*n_cols + j)]*sobel_kernel_x[1][2];
			xval += src_ptr[((i+1)*n_cols + j + 1)]*sobel_kernel_x[2][2];
			yval += src_ptr[((i+1)*n_cols + j + 1)]*sobel_kernel_x[2][2];

			//cap at 255 (uint8 max val)
			uint32_t newval = (uint32_t) round(sqrt((xval*xval)+(yval*yval)));
			if(newval > 255)
			{
				dst_ptr[i*n_cols + j] = 255;
			}
			else
			{
				dst_ptr[i*n_cols + j] = (uint8_t)newval;
			}
		}
	}
}

void sobel_neon(const Mat& src, Mat& dst)
{
	const uint8_t* src_ptr = src.ptr<uint8_t>();
    uint8_t* dst_ptr = dst.ptr<uint8_t>();

	int n_cols = src.cols;
	int n_rows = src.rows;

	//create neon sobel vectors for each dimension and load data
	int16x8_t sobel_vec_x;
	int16x8_t sobel_vec_y;
	sobel_vec_x = vld1q_s16(sobel_kernel_x_vec);
	sobel_vec_y = vld1q_s16(sobel_kernel_y_vec);

	for(int i = 1; i < (n_rows-1); i++)
	{
		int j = 1;
		while(j < (n_cols-1))
		{
			//make data vector for current position
			int16_t src_data[8];
			src_data[0] = src_ptr[((i-1)*n_cols + j - 1)];
			src_data[1] = src_ptr[((i-1)*n_cols + j)];
			src_data[2] = src_ptr[((i-1)*n_cols + j + 1)];
			src_data[3] = src_ptr[((i)*n_cols + j - 1)];
			src_data[4] = src_ptr[((i)*n_cols + j + 1)];
			src_data[5] = src_ptr[((i+1)*n_cols + j - 1)];
			src_data[6] = src_ptr[((i+1)*n_cols + j)];
			src_data[7] = src_ptr[((i+1)*n_cols + j + 1)];
			int16x8_t src_data_vec;
			src_data_vec = vld1q_s16(src_data);

			//initialize result vectors
			int16x8_t x_result_vec;
			int16x8_t y_result_vec;
			x_result_vec = vmovq_n_s16(0);
			y_result_vec = vmovq_n_s16(0);

			//multiply-accumulate
			x_result_vec = vmlaq_s16(x_result_vec,src_data_vec,sobel_vec_x);
			y_result_vec = vmlaq_s16(y_result_vec,src_data_vec,sobel_vec_y);

			//sum across vectors
			int16_t x_sum = vaddvq_s16(x_result_vec);
			int16_t y_sum = vaddvq_s16(y_result_vec);

			//cap at 255 (uint8 max val)
			uint32_t result = (uint32_t) round(sqrt((x_sum*x_sum)+(y_sum*y_sum)));
			if(result > 255)
			{
				dst_ptr[i*n_cols + j] = 255;
			}
			else
			{
				dst_ptr[i*n_cols + j] = (uint8_t)result;
			}
			j++;
		}
	}
}

