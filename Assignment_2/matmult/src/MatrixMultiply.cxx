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
