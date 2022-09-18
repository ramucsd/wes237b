#include <iostream>
#include <math.h>

#include "sigmoid_activation.h"
#include "nn_exception.h"

using namespace std;

__device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

__global__ void sigmoidActivationForward(float* input, float* output, const int input_rows, const int input_cols)
{
    //TODO: complete the sigmoid activation for forward propagation
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (input_rows*input_cols))
	{
		output[idx] = sigmoid(input[idx]);
	}

}

__global__ void sigmoidActivationBackprop(float* input, float* errorFromLayerBelow, float* errorToLayerAbove, const int input_rows, const int input_cols)
{
    //TODO: complete the derivative of the sigmoid activation for back propagation. This function returns the error to go to the layer above (eA)

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (input_rows*input_cols))
	{
		errorToLayerAbove[idx] = errorFromLayerBelow[idx] * sigmoid(input[idx]) * (1.0 - sigmoid(input[idx]));
	}
	
}

SigmoidActivation::SigmoidActivation(string name)
{
	this->name = name;
}

SigmoidActivation::~SigmoidActivation()
{ }

Matrix& SigmoidActivation::forward(Matrix& input)
{
	this->input = input;

	output.allocateMemoryIfNotAllocated(input.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((input.shape.rows * input.shape.cols + block_size.x - 1) / block_size.x);
	
	sigmoidActivationForward<<<num_of_blocks, block_size>>>(input.data_device,
																output.data_device,
																input.shape.rows, input.shape.cols);

	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");

	return output;
}

Matrix& SigmoidActivation::backprop(Matrix& errorFromLayerBelow, float learning_rate)
{
	errorToLayerAbove.allocateMemoryIfNotAllocated(input.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((input.shape.rows * input.shape.cols + block_size.x - 1) / block_size.x);
	
	sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(input.data_device,
																errorFromLayerBelow.data_device,
																errorToLayerAbove.data_device,
																input.shape.rows, input.shape.cols);

	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation.");

	return errorToLayerAbove;
}