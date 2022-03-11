#ifndef _LU_H
#define _LU_H

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define N 64

#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

void AB_1(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N]);

#endif
