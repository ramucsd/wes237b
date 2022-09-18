#include "mse_cost.h"
#include "nn_exception.h"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void meanSquaredErrorCost(float* predictions, float* target, const int size, float* cost)
{
    //TODO: complete the cost calculation. (Hint: use atomicAdd() when accumulating)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		float e = predictions[idx] - target[idx];
		e = e*e/size;
		atomicAdd(cost,e);
	}
}

__global__ void dMeanSquaredErrorCost(float* predictions, float* target, float* eA, const int size)
{
    //TODO: complete the derivative of the mean squared error cost function
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		eA[idx] = 2.0*(predictions[idx] - target[idx]);
	}
}

float MSECost::cost(Matrix predictions, Matrix target)
{
	assert(predictions.shape.cols == target.shape.cols);

	float* cost;
	cudaMallocManaged(&cost, sizeof(float));

	*cost = 0.0f;

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.cols + block_size.x - 1)/ block_size.x);

	meanSquaredErrorCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device, predictions.shape.cols, cost);

	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorsOccurred("Cannot compute mean squared error cross entropy cost.");

	float cost_value = *cost;

	cudaFree(cost);

	return cost_value;
}

Matrix MSECost::dCost(Matrix predictions, Matrix target, Matrix eA)
{
	assert(predictions.shape.cols == target.shape.cols);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.cols + block_size.x - 1) / block_size.x);

	dMeanSquaredErrorCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device, eA.data_device, predictions.shape.cols);

	return eA;
}
