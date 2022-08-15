#include "main.h"
#include "MatrixMultiply.h"

using namespace cv;

#define USE_LUT //comment out to not use Lookup table

Mat LUT_w;
Mat LUT_h;


// Helper function
float sf(int in){
	if (in == 0)
		return 0.70710678118; // = 1 / sqrt(2)
	return 1.;
}

// Initialize LUT
void initDCT(int WIDTH, int HEIGHT)
{
    LUT_w = Mat(HEIGHT, WIDTH, CV_32FC1);
    LUT_h = Mat(HEIGHT, WIDTH, CV_32FC1);

    float* w_ptr = LUT_w.ptr<float>();
    float* h_ptr  = LUT_h.ptr<float>();

    //split scale factors so it works for matrix multiplication
    float scale_w = sqrt(2./(float)WIDTH);
    float scale_h = sqrt(2./(float)HEIGHT);

    for(int y = 0; y < WIDTH; y++)
    {
        for(int j = 0; j < WIDTH; j++)
        {
            w_ptr[y * WIDTH + j] = scale_w*sf(y)*cos(M_PI/((float)WIDTH)*(j+1./2.)*(float)y);
        }
    }
    for(int x = 0; x < HEIGHT; x++)
    {
        for(int i = 0; i < HEIGHT; i++)
        {
            h_ptr[x * HEIGHT + i] = scale_h*sf(x)*cos(M_PI/((float)HEIGHT)*(i+1./2.)*(float)x);
        }
    }
}

// Baseline: O(N^4)
Mat student_dct_naive(Mat input)
{
	const int HEIGHT = input.rows;
	const int WIDTH  = input.cols;

	float scale = 2./sqrt(HEIGHT*WIDTH);

	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

	// Note: Using pointers is faster than Mat.at<float>(x,y)
	// Try to use pointers for your LUT as well
	float* result_ptr = result.ptr<float>();
	float* input_ptr  = input.ptr<float>();
    float* w_ptr = LUT_w.ptr<float>();
    float* h_ptr  = LUT_h.ptr<float>();

	for(int x = 0; x < HEIGHT; x++)
	{
		for(int y = 0; y < WIDTH; y++)
		{
			float value = 0.f;

			for(int i = 0; i < HEIGHT; i++)
			{
				for(int j = 0; j < WIDTH; j++)
				{
                    #ifdef USE_LUT
                        value += input_ptr[i*WIDTH+j]*w_ptr[y*WIDTH+j]*h_ptr[x*HEIGHT+i];
                    #else
                        value += input_ptr[i * WIDTH + j]
						* cos(M_PI/((float)HEIGHT)*(i+1./2.)*(float)x)
						* cos(M_PI/((float)WIDTH)*(j+1./2.)*(float)y);
                    #endif
				}
			}
			#ifndef USE_LUT
                value = scale * sf(x) * sf(y) * value;
            #endif

			result_ptr[x * WIDTH + y] = value;
		}
	}

	return result;
}

Mat student_dct_opt(Mat input){
	const int HEIGHT = input.rows;
	const int WIDTH = input.cols;

	float scale = 2./sqrt(HEIGHT*WIDTH);

	// Create the result matrix of the correct datatype
	Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);
	Mat result_row = Mat(HEIGHT, WIDTH, CV_32FC1);

	float* result_ptr = result.ptr<float>();
    float* result_row_ptr = result_row.ptr<float>();
	float* input_ptr = input.ptr<float>();
    float* w_ptr = LUT_w.ptr<float>();
    float* h_ptr  = LUT_h.ptr<float>();

	// Less naive implementation.
	// Perform 2 1D DCTs, one for the rows and one for the columns
	float value;
	for(int k=0; k<HEIGHT; k++) {
		for(int i=0; i<WIDTH; i++) {
			value = 0.0;
			for(int j=0; j<WIDTH; j++) {
                #ifdef USE_LUT
                    value += input_ptr[k*WIDTH+j]*w_ptr[i*WIDTH+j];
                #else
                    value += input.at<float>(k, j) * cos(M_PI/((float)WIDTH)*(j+1./2.)*(float)i);
                #endif
			}
            #ifdef USE_LUT
                result_row_ptr[k*HEIGHT+i] = value;
            #else
                result_row.at<float>(k,i) = value * sf(i);
            #endif
		}
	}

	// Now perform the column transformation
	for(int k=0; k<WIDTH; k++) {
		for(int i=0; i<HEIGHT; i++) {
			value = 0.0;
			for (int j=0; j<HEIGHT; j++) {
                #ifdef USE_LUT
                    value += result_row_ptr[j*HEIGHT+k]*h_ptr[i*HEIGHT+j];
                #else
                    value += result_row.at<float>(j,k) * cos(M_PI/((float)HEIGHT)*(j+1./2.)*(float)i);
                #endif
			}
            #ifdef USE_LUT
                result_ptr[i*WIDTH+k] = value;
            #else
                result.at<float>(i, k) = value*sf(i)*scale;
            #endif
		}
	}

	return result;
}


// *****************
//   Hint
// *****************
//
// DCT as matrix multiplication


Mat student_dct_MM(Mat input)
{
	// -- Works only for WIDTH == HEIGHT
	assert(input.rows == input.cols);
    
    // -- Matrix multiply with my code from Part 2
    //initialize required matrices
    Mat output = Mat(input.rows, input.rows, CV_32FC1);
    Mat temp = Mat(input.rows, input.rows, CV_32FC1);
    Mat LUT_t = LUT_w.t();
       
    //convert to float pointers (since that's what my function takes)
    float* input_ptr = input.ptr<float>();
    float* LUT_ptr = LUT_w.ptr<float>();
    float* temp_ptr = temp.ptr<float>();
    float* LUT_t_ptr = LUT_t.ptr<float>();
    float* output_ptr = output.ptr<float>();
    
    //perform the multiplies
    matMult(LUT_ptr,input_ptr,temp_ptr,input.rows);
    matMult(temp_ptr,LUT_t_ptr,output_ptr,input.rows);

	return output;
}



//DCT using Block Matrix Multiplication
Mat student_dct_BMM(Mat input)
{
	// -- Works only for WIDTH == HEIGHT
	assert(input.rows == input.cols);
    
    // -- Block Matrix multiply
    //initialize required matrices
    Mat output = Mat(input.rows, input.rows, CV_32FC1);
    Mat temp = Mat(input.rows, input.rows, CV_32FC1);
    Mat LUT_t = LUT_w.t();
       
    //convert to float pointers (since that's what my function takes)
    float* input_ptr = input.ptr<float>();
    float* LUT_ptr = LUT_w.ptr<float>();
    float* temp_ptr = temp.ptr<float>();
    float* LUT_t_ptr = LUT_t.ptr<float>();
    float* output_ptr = output.ptr<float>();
    
    //perform the multiplies
    matMultBlock(LUT_ptr,input_ptr,temp_ptr,input.rows);
    matMultBlock(temp_ptr,LUT_t_ptr,output_ptr,input.rows);
    

	return output;
}
