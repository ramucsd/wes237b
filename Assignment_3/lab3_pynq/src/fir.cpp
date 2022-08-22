#include "main.h"
#include "arm_neon.h"

// ----------------------------------------------
// Run a FIR filter on the given input data
// ----------------------------------------------
void fir(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    //TODO
    for(int i = 0; i < length - filterLength; i++)
    {
        float val = 0;
        for(int j = 0; j < filterLength; j++)
        {
                val += coeffs[j]*input[i+j];
        }
        output[i] = val;
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using Loop Unrolling
// ----------------------------------------------
void fir_opt(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    //TODO
    for(int i = 0; i < length - filterLength; i++)
    {
        float val = 0;
        for(int j = 0; j < filterLength; j+=4)
        {
            val += coeffs[j]*input[i+j];
            val += coeffs[j+1]*input[i+j+1];
            val += coeffs[j+2]*input[i+j+2];
            val += coeffs[j+3]*input[i+j+3];
        }
        output[i] = val;
    }
}

// ----------------------------------------------
// Run a FIR filter on the given input data using NEON
// ----------------------------------------------
void fir_neon(float *coeffs, float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
    //TODO
    for(int i = 0; i < length - filterLength; i++)
    {
        float32x4_t coeff_vec;
        float32x4_t input_vec;
        float32x4_t partials_vec;
        partials_vec = vmovq_n_f32(0);
        
        for(int j = 0; j < filterLength; j+=4)
        {
            coeff_vec = vld1q_f32(&coeffs[j]);
            input_vec = vld1q_f32(&input[i+j]);
            partials_vec = vmlaq_f32(partials_vec,coeff_vec,input_vec);
        }
        float partials[4];
        vst1q_f32(&partials[0],partials_vec);
        output[i] = partials[0] + partials[1] + partials[2] + partials[3];
        
    }
}


// ----------------------------------------------
// Create filter coefficients
// ----------------------------------------------
void designLPF(float* coeffs, int filterLength, float Fs, float Fx)
// ----------------------------------------------
{
	float lambda = M_PI * Fx / (Fs/2);

	for(int n = 0; n < filterLength; n++)
	{
		float mm = n - (filterLength - 1.0) / 2.0;
		if( mm == 0.0 ) coeffs[n] = lambda / M_PI;
		else coeffs[n] = sin( mm * lambda ) / (mm * M_PI);
	}
}
