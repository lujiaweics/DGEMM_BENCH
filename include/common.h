#ifndef COMMON_H_
#define COMMON_H_

#define MR 8
#define min(x, y) ((x) < (y) ? x : y)
#define alpha( i,j ) A[ (j)*ldA + (i) ]
#define beta( i,j )  B[ (j)*ldB + (i) ]
#define gamma(i, j) C[(j) * ldC + (i)]
#define A_(i, j) A_ptr[(j) * ldA + (i)]
#define B_(i, j) B_ptr[(j) * ldB + (i)]
#define C_(i, j) C_ptr[(j) * ldC + (i)]

#endif