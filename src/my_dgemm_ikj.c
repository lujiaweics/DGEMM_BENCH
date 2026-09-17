/*
 * my_dgemm_ijk.c -- Naive ijk triple-loop implementation (row-major)
 *
 * Computes C := alpha * A * B + beta * C
 *   A : m x k, leading dimension ldA
 *   B : k x n, leading dimension ldB
 *   C : m x n, leading dimension ldC
 *
 * This is the "performance floor" version of the entire benchmark and the
 * baseline for all subsequent optimizations (ikj / cache blocking / register
 * blocking / packing / multi-threading). Its performance deficiencies are
 * very typical:
 *
 *   1) The inner loop walks along the p dimension. Row i of A is contiguous
 *      (good), but elements of B are accessed across rows with stride ldB
 *      (bad). Each iteration requires a new cache line, causing B to be
 *      repeatedly fetched from memory/cache.
 *   2) For each (i,j) combination, the entire row i of A and all rows of B
 *      must be scanned. Data reuse relies entirely on the cache, resulting
 *      in very low arithmetic intensity.
 *   3) The loop body has few independent multiply-add operations, making it
 *      difficult for the compiler to perform effective vectorization.
 *
 * Accumulating the k-dimensional dot product in a local variable 'sum'
 * before writing back to C allows the compiler to keep the accumulator in
 * a register, avoiding repeated reads/writes of C(i,j). This is also the
 * standard form of the "naive implementation".
 */
#include "dgemm_bench.h"

/* Row-major element mapping: X(i,j) == X_ptr[(i)*ldX + (j)] */
#define A_(i, j) A_ptr[(i) * ldA + (j)]
#define B_(i, j) B_ptr[(i) * ldB + (j)]
#define C_(i, j) C_ptr[(i) * ldC + (j)]

void my_dgemm(int m, int n, int k,
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

  for (int i = 0; i < m; i++)
  {
    for (int p = 0; p < k; p++)
    {
      for (int j = 0; j < n; j++)
      {
        C_(i, j) += alpha * A_(i, p) * B_(p, j);
      }
    }
  }
}
