#include "dgemm_bench.h"
#include "micro_kernel.h"

void my_dgemm_colmaj(int m, int n, int k,
                     double alpha, const double *A_ptr, int ldA,
                     const double *B_ptr, int ldB,
                     double beta, double *C_ptr, int ldC);

/* Row-major element mapping: X(i,j) == X_ptr[(i)*ldX + (j)] */
#define A_(i, j) A_ptr[(j) * ldA + (i)]
#define B_(i, j) B_ptr[(j) * ldB + (i)]
#define C_(i, j) C_ptr[(j) * ldC + (i)]

void my_dgemm(int m, int n, int k,
              double alpha, const double *A_ptr, int ldA,
              const double *B_ptr, int ldB,
              double beta, double *C_ptr, int ldC)
{
  my_dgemm_colmaj(n, m, k, alpha, B_ptr, ldB, A_ptr, ldA, beta, C_ptr, ldC);
}

void my_dgemm_colmaj(int m, int n, int k,
                     double alpha, const double *A_ptr, int ldA,
                     const double *B_ptr, int ldB,
                     double beta, double *C_ptr, int ldC)
{
  if (beta != 1.0)
  {
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
      {
        C_(i, j) *= beta;
      }
    }
  }

  for (int j = 0; j < n; j+=NR)
  {
    for (int i = 0; i < m; i+=MR)
    {
      Micro_kernel(k, &A_(i, 0), ldA, &B_(0, j), ldB, &C_(i, j), ldC);
    }
  }
}
