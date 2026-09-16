/*
 * max_abs_diff.c -- Maximum Absolute Difference Between Two Matrices
 *
 * This is the correctness criterion for the entire benchmark: the tested
 * implementation result C is compared element-wise with the OpenBLAS reference
 * result Cref, and the maximum absolute difference is taken. Since both are
 * accumulated from the same initial C value, this value should normally be on
 * the order of floating-point roundoff (approximately 1e-13 or smaller).
 */
#include "dgemm_bench.h"

/* Row-major element mapping */
#define X_( i, j )  x[(i) * ldx + (j)]
#define Y_( i, j )  y[(i) * ldy + (j)]

double max_abs_diff( int m, int n,
                     const double *x, int ldx,
                     const double *y, int ldy )
{
  double diff = 0.0;

  for ( int i = 0; i < m; i++ ) {
    for ( int j = 0; j < n; j++ ) {
      double d = X_( i, j ) - Y_( i, j );

      if ( d < 0.0 )
        d = -d;

      if ( d > diff )
        diff = d;
    }
  }

  return diff;
}
