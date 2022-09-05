#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "matrixmul.h"
#include "timer.h"

#define BLOCK_SIZE 16

__global__ void block_mm_kernel(const float* A, const float* B, float* output, int M, int N) 
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int xb = blockDim.x*blockIdx.x;
	int yb = blockDim.y*blockIdx.y;

	__shared__ float ab[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bb[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float cb[BLOCK_SIZE][BLOCK_SIZE];

	//printf("block[%d,%d], thread[%d,%d]\n", blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x);
	cb[y][x] = 0;
    
	//loop through the blocks
	for (int zb = 0; zb < M; zb+= BLOCK_SIZE)
        {

		ab[y][x] = A[(yb+y)*M + zb + x];
		bb[y][x] = B[(zb+y)*N + xb + x];

		__syncthreads();

        //perform the matrix multiply on the blocks
		for(int i = 0; i < BLOCK_SIZE; i++)
		{
			cb[y][x] += ab[y][i]*bb[i][x];
		}
                
		__syncthreads();
	}
	output[(yb+y)*N + xb + x] = cb[y][x];
}


inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}



float run_mm_gpu(const float* A, const float* B, float* C, int M, int N)
{

	if ((N % BLOCK_SIZE != 0) || (M % BLOCK_SIZE != 0))
    	{
        	std::cout << "ERROR: input matrix size (" << N << "x" << M << ") not divisible by block size (" << BLOCK_SIZE << "x" << BLOCK_SIZE << ")." << std::endl;
        	std::cout << "Change the input size or use a different calculation algorithm." << std::endl;
        	return -99.9;
    	}
	Timer gpu_timer;
	gpu_timer.start();

	dim3 t(BLOCK_SIZE,BLOCK_SIZE);
	int nb = N/BLOCK_SIZE;
	dim3 b(nb,nb);
	block_mm_kernel<<<b,t>>>(A,B,C,M,N);
	
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}


