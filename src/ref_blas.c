/*
 * ref_blas.c -- Reference (golden) implementation: OpenBLAS / CBLAS
 *
 * Isolating BLAS calls provides two benefits:
 *   1) driver.c does not need to include <cblas.h> or depend on specific BLAS
 *      header paths;
 *   2) To switch the reference implementation to Fortran ABI dgemm_ or other BLAS
 *      libraries (MKL / BLIS / ATLAS), only this file needs modification.
 *
 * CBLAS calling convention:
 *   cblas_dgemm( layout, transA, transB, m, n, k,
 *                alpha, A, ldA, B, ldB, beta, C, ldC )
 * Corresponding computation: C := alpha * op(A) * op(B) + beta * C
 * Here op = NoTrans, and all matrices are row-major, consistent with my_dgemm semantics.
 */
#include <cblas.h>

#include "dgemm_bench.h"

void ref_set_threads(int threads) {
       openblas_set_num_threads(threads);
}

void ref_dgemm( int m, int n, int k,
                double alpha, const double *A, int ldA,
                              const double *B, int ldB,
                double beta,  double *C, int ldC )
{
  cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
               m, n, k,
               alpha, A, ldA,
                      B, ldB,
               beta,  C, ldC );
}
