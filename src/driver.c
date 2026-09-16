/*
 * driver.c -- DGEMM Performance and Correctness Test Driver
 *
 * Responsibilities (decoupled from specific algorithms, no changes needed
 * when adding new implementations):
 *   1) Generate random test data and prepare identical initial values for both
 *      the tested implementation and the reference implementation;
 *   2) Time separately: OpenBLAS reference implementation (ref_dgemm) and
 *      the tested implementation (my_dgemm);
 *   3) Repeat each size nrepeats times, use the best (fastest) result;
 *   4) Verify consistency between tested and reference results using max_abs_diff;
 *   5) Write results to stdout as CSV (lines starting with "#" are metadata
 *      comments), redirected by Makefile to data/output_<impl>.csv for
 *      Python3 plotting scripts.
 *
 * Usage:
 *   ./dgemm_<impl>.x [nrepeats] [first] [last] [inc]
 *   Default values for the four parameters are 3 48 1500 48 (consistent with
 *   Makefile variables).
 *
 * Output format:
 *   # implementation,<name>
 *   # repeats,<count>
 *   # sizes,first=..,last=..,inc=..
 *   n,ref_time_s,ref_gflops,my_time_s,my_gflops,diff
 *   48,1.23e-05,1.79e+01,8.76e-05,2.51e+00,3.55e-15
 *   ...
 *
 * Note: Matrix sizes are traversed from large to small. Timing for large
 * matrices is less affected by noise. Running large matrices first stabilizes
 * subsequent measurements for smaller matrices (this approach follows the
 * design from LAFF-On-PfHP).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dgemm_bench.h"

/* Injected by Makefile via -DIMPL_NAME='"ijk"' to identify current implementation */
#ifndef IMPL_NAME
#define IMPL_NAME "unknown"
#endif

int main( int argc, char *argv[] )
{
  ref_set_threads(1);
  
  int    m, n, k, ldA, ldB, ldC;
  int    size, first, last, inc, irep, nrepeats;
  double d_one = 1.0;
  double dtime, dtime_best, diff, maxdiff = 0.0, gflops;
  double *A, *B, *C, *Cold, *Cref;

  /* ---------------- 1) Parse experiment parameters ---------------- */
  nrepeats = ( argc > 1 ) ? atoi( argv[1] ) : 3;
  first    = ( argc > 2 ) ? atoi( argv[2] ) : 48;
  last     = ( argc > 3 ) ? atoi( argv[3] ) : 1500;
  inc      = ( argc > 4 ) ? atoi( argv[4] ) : 48;

  if ( nrepeats < 1 )
    nrepeats = 1;
  if ( inc <= 0 )
    inc = 48;

  /* Align first / last to integer multiples of inc */
  last  = ( last  / inc ) * inc;
  first = ( first / inc ) * inc;
  if ( first == 0 )
    first = inc;
  if ( last < first )
    last = first;

  /* Fixed random seed: ensures identical test data across runs, reproducible results */
  srand( 0 );

  printf( "# implementation,%s\n", IMPL_NAME );
  printf( "# repeats,%d\n", nrepeats );
  printf( "# sizes,first=%d,last=%d,inc=%d\n", first, last, inc );
  printf( "n,ref_time_s,ref_gflops,my_time_s,my_gflops,diff\n" );

  /* ---------------- 2) Run experiments for each size ---------------- */
  for ( size = last; size >= first; size -= inc ) {

    /* This benchmark only tests square matrices */
    m = n = k = size;
    ldA = ldB = ldC = size;   /* Row-major: leading dimension = number of columns */

    /* Floating point operations per GEMM: one multiply-add per element, counted as 2 FLOPs */
    gflops = 2.0 * ( double ) m * n * k * 1.0e-09;

    /* A: m x k, B: k x n, C: m x n */
    A    = ( double * ) malloc( ( size_t ) ldA * m * sizeof( double ) );
    B    = ( double * ) malloc( ( size_t ) ldB * k * sizeof( double ) );
    C    = ( double * ) malloc( ( size_t ) ldC * m * sizeof( double ) );
    Cold = ( double * ) malloc( ( size_t ) ldC * m * sizeof( double ) );
    Cref = ( double * ) malloc( ( size_t ) ldC * m * sizeof( double ) );

    if ( A == NULL || B == NULL || C == NULL || Cold == NULL || Cref == NULL ) {
      fprintf( stderr, "malloc failed at n = %d\n", size );
      return 1;
    }

    random_matrix( m, k, A,    ldA );
    random_matrix( k, n, B,    ldB );
    random_matrix( m, n, Cold, ldC );

    /* ---------- 2.1 Reference implementation: OpenBLAS ---------- */
    for ( irep = 0; irep < nrepeats; irep++ ) {
      /* GEMM is an accumulation operation; C must be reset to initial value each round */
      memcpy( Cref, Cold, ( size_t ) ldC * m * sizeof( double ) );

      dtime = clock_seconds();
      ref_dgemm( m, n, k, d_one, A, ldA, B, ldB, d_one, Cref, ldC );
      dtime = clock_seconds() - dtime;

      if ( irep == 0 )
        dtime_best = dtime;
      else
        dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    printf( "%5d,%12.5le,%12.5le,", n, dtime_best, gflops / dtime_best );
    /* Flush promptly to avoid output buffering affecting subsequent timing */
    fflush( stdout );

    /* ---------- 2.2 Tested implementation ---------- */
    for ( irep = 0; irep < nrepeats; irep++ ) {
      memcpy( C, Cold, ( size_t ) ldC * m * sizeof( double ) );

      dtime = clock_seconds();
      my_dgemm( m, n, k, d_one, A, ldA, B, ldB, d_one, C, ldC );
      dtime = clock_seconds() - dtime;

      if ( irep == 0 )
        dtime_best = dtime;
      else
        dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    /* ---------- 2.3 Correctness verification ---------- */
    diff    = max_abs_diff( m, n, C, ldC, Cref, ldC );
    maxdiff = ( diff > maxdiff ? diff : maxdiff );

    printf( "%12.5le,%12.5le,%12.5le\n", dtime_best, gflops / dtime_best, diff );
    fflush( stdout );

    free( A );
    free( B );
    free( C );
    free( Cold );
    free( Cref );
  }

  /* Summary info goes to stderr to avoid polluting stdout CSV */
  fprintf( stderr, "# max |diff| over all sizes = %le\n", maxdiff );

  return 0;
}
