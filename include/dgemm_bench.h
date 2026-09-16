/*
 * dgemm_bench.h -- DGEMM-Bench Public Interface Declaration
 *
 * Project-wide conventions:
 *   1) All matrices are stored as 1D double arrays in row-major order.
 *      Element mapping: X(i,j) == X_ptr[(i)*ldX + (j)],
 *      where leading dimension ldX >= number of columns.
 *   2) All tested and reference implementations share the same my_dgemm / ref_dgemm
 *      signatures. The driver only depends on these two symbols, so adding a new
 *      implementation only requires providing a function with the same signature
 *      and re-linking.
 */
#ifndef DGEMM_BENCH_H
#define DGEMM_BENCH_H

/* ---------------------------------------------------------------
 * Timing
 * Returns seconds since the first call in this process (monotonic clock,
 * unaffected by system time adjustments).
 * Usage: t0 = clock_seconds(); ... ; elapsed = clock_seconds() - t0;
 * --------------------------------------------------------------- */
double clock_seconds( void );

/* ---------------------------------------------------------------
 * Test Data
 * Fill an m x n matrix with uniform random numbers in [0,1).
 * Random seed is controlled by the caller via srand() (fixed seed in driver
 * for reproducibility).
 * --------------------------------------------------------------- */
void random_matrix( int m, int n, double *x, int ldx );

/* ---------------------------------------------------------------
 * Result Verification
 * Returns the maximum absolute difference between corresponding elements
 * of two m x n matrices (row-major, with leading dimensions ldx / ldy).
 * Used for numerical consistency comparison with reference implementation.
 * --------------------------------------------------------------- */
double max_abs_diff( int m, int n,
                     const double *x, int ldx,
                     const double *y, int ldy );

/* ---------------------------------------------------------------
 * DGEMM Unified Interface (Pluggable Algorithm Layer)
 *
 * Semantics: C := alpha * A * B + beta * C
 *   A : m x k, leading dimension ldA
 *   B : k x n, leading dimension ldB
 *   C : m x n, leading dimension ldC
 *
 * Each implementation file (src/my_dgemm_*.c) must export this function.
 * The Makefile determines which one to link.
 * --------------------------------------------------------------- */
void my_dgemm( int m, int n, int k,
               double alpha, const double *A, int ldA,
                             const double *B, int ldB,
               double beta,  double *C, int ldC );

/* ---------------------------------------------------------------
 * Reference (golden) implementation: OpenBLAS CBLAS interface with the same
 * semantics as my_dgemm. Encapsulated in src/ref_blas.c so the driver does
 * not directly depend on cblas.h. If switching to Fortran ABI dgemm_ or other
 * BLAS libraries, only that file needs modification.
 * --------------------------------------------------------------- */
void ref_dgemm( int m, int n, int k,
                double alpha, const double *A, int ldA,
                              const double *B, int ldB,
                double beta,  double *C, int ldC );

void ref_set_threads(int threads);

#endif /* DGEMM_BENCH_H */
