#ifndef MATRIXMULTIPLY_H
#define MATRIXMULTIPLY_H

#include <math.h>

// Calculates the product of two identically sized square matrices
//  @param      *a Pointer to first matrix
//  @param      *b Pointer to second matrix
//  @param *result Pointer to result matrix
//  @param    size The size of each dimension of each matrix (assumed to be square)
void matMult(float *a, float *b, float *result, int size);

#endif
