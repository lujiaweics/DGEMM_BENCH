/*
 * random_matrix.c -- Fill a matrix with random numbers
 *
 * This implementation deliberately uses the standard library rand() instead of
 * POSIX drand48() to ensure direct compilation on Linux / macOS / Windows (MinGW).
 * Note that rand() has lower numerical quality, but for performance benchmarks,
 * data distribution does not affect memory access patterns, so it is acceptable.
 * The random seed is controlled by the caller.
 */
#include <stdlib.h>

#include "dgemm_bench.h"

/* Row-major element mapping: X(i,j) == x[(i)*ldx + (j)] */
#define X_( i, j )  x[(i) * ldx + (j)]

void random_matrix( int m, int n, double *x, int ldx )
{
  const double scale = 1.0 / ( ( double ) RAND_MAX + 1.0 );

  for ( int i = 0; i < m; i++ )
    for ( int j = 0; j < n; j++ )
      X_( i, j ) = ( double ) rand() * scale;   /* Value in [0,1) */
}
