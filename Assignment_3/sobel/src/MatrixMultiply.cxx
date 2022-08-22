#include <iostream>
#include <stdio.h>
#include <math.h>
#include "MatrixMultiply.h"



void matMult(float *a, float *b, float *result, int size) {
    for (int x = 0; x < size; x++)
    {
        for (int y = 0; y < size; y++)
        {
            float val = 0;
            for (int z = 0; z < size; z++)
            {
                //val += a[x][z]*b[z][y];
                val += a[x*size + z]*b[z*size + y];
            }
            result[x*size + y] = val;
        }
    }
}

void matMultBlock(float *a, float *b, float *result, int size) {
    if (size % BLOCK_SIZE != 0)
    {
        std::cout << "ERROR: input matrix size (" << size << ") not divisible by block size (" << BLOCK_SIZE << ").\n";
        std::cout << "Change the input size or use a different calculation algorithm." << std::endl;
        return;
    }
    
    //loop through the blocks
    for (int xb = 0; xb < size; xb+= BLOCK_SIZE)
    {
        for (int yb = 0; yb < size; yb+= BLOCK_SIZE)
        {
            for (int zb = 0; zb < size; zb+= BLOCK_SIZE)
            {
                //allocate smaller matrices so they can fit in cache
                float ab[BLOCK_SIZE][BLOCK_SIZE] = {};
                float bb[BLOCK_SIZE][BLOCK_SIZE] = {};
                float rb[BLOCK_SIZE][BLOCK_SIZE] = {};
                
                
                for (int y = 0; y < BLOCK_SIZE; y++)
                {
                    for (int x = 0; x < BLOCK_SIZE; x++)
                    {
                        ab[y][x] = a[(yb+y)*size + zb + x];
                    }
                }
                
                for (int y = 0; y < BLOCK_SIZE; y++)
                {
                    for (int x = 0; x < BLOCK_SIZE; x++)
                    {
                        bb[y][x] = b[(zb+y)*size + xb + x];
                    }
                }
                
                //perform the matrix multiply on the blocks
                matMult(&ab[0][0],&bb[0][0],&rb[0][0],BLOCK_SIZE);
                
                for (int y = 0; y < BLOCK_SIZE; y++)
                {
                    for (int x = 0; x < BLOCK_SIZE; x++)
                    {
                        //sum the result of each block used for each output
                        //need to initialize to 0 first, so zb = 0 does not use sum
                        if (zb == 0)
                        {
                            result[(yb+y)*size + xb + x] = rb[y][x];
                        }
                        else
                        {
                            result[(yb+y)*size + xb + x] += rb[y][x];
                        }
                    }
                }
            } 
        }
    }
}