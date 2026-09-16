/*
 * timer.c -- Cross-platform monotonic high-resolution timer
 *
 * Design considerations:
 *   1) Use monotonic clock to avoid jumps or negative durations when system time
 *      is adjusted;
 *   2) Record a reference time on first call; subsequent returns are "seconds
 *      since reference". Thus, elapsed time is obtained by subtraction, without
 *      caller concern for absolute time;
 *   3) The _POSIX_C_SOURCE feature macro is defined inside this file and must
 *      precede any system headers. Reason: when compiling with -std=c11, the
 *      compiler is in strict ISO mode; without explicitly enabling POSIX
 *      extensions, glibc will not declare clock_gettime, and CLOCK_MONOTONIC
 *      will not be visible (resulting in implicit declaration of function
 *      'clock_gettime' and 'CLOCK_MONOTONIC' undeclared). Defining this in the
 *      file ensures it does not depend on Makefile options, and will work when
 *      compiled standalone or with different build systems;
 *   4) Platform branches:
 *        Windows  : QueryPerformanceCounter / QueryPerformanceFrequency
 *        macOS    : mach_absolute_time + mach_timebase_info
 *        Other POSIX: clock_gettime( CLOCK_MONOTONIC )
 *      The Windows branch is necessary because MinGW/MSVC do not guarantee
 *      clock_gettime availability.
 */

#if defined( _WIN32 ) || defined( __WIN32__ ) || defined( _WIN64 )

/* ======================= Windows ======================= */

#include <windows.h>

static double reference_seconds = 0.0;

double clock_seconds( void )
{
  static LARGE_INTEGER frequency;
  static int frequency_ready = 0;
  LARGE_INTEGER counter;

  if ( !frequency_ready ) {
    QueryPerformanceFrequency( &frequency );   /* Counter frequency, typically 1e7 Hz */
    frequency_ready = 1;
  }

  QueryPerformanceCounter( &counter );

  double now = ( double ) counter.QuadPart / ( double ) frequency.QuadPart;

  if ( reference_seconds == 0.0 )
    reference_seconds = now;

  return now - reference_seconds;
}

#else   /* ==================== Non-Windows ==================== */

/* Must be defined before including any system headers, otherwise clock_gettime will not be declared */
#ifndef _POSIX_C_SOURCE
  #define _POSIX_C_SOURCE 200809L
#endif

#if defined( __APPLE__ ) || defined( __MACH__ )
  #include <mach/mach_time.h>
#else
  #include <time.h>
#endif

static double reference_seconds = 0.0;

#if defined( __APPLE__ ) || defined( __MACH__ )

double clock_seconds( void )
{
  static mach_timebase_info_data_t timebase;
  static int timebase_ready = 0;

  if ( !timebase_ready ) {
    mach_timebase_info( &timebase );   /* nanoseconds = count * numer / denom */
    timebase_ready = 1;
  }

  double now = ( double ) mach_absolute_time() * 1.0e-9
               * ( double ) timebase.numer / ( double ) timebase.denom;

  if ( reference_seconds == 0.0 )
    reference_seconds = now;

  return now - reference_seconds;
}

#else   /* Other POSIX (Linux / WSL / MSYS2 / Cygwin etc.) */

double clock_seconds( void )
{
  struct timespec ts;

  #ifdef CLOCK_MONOTONIC
    clock_gettime( CLOCK_MONOTONIC, &ts );
  #else
    /* Fallback: rare platforms without CLOCK_MONOTONIC, degrade to real-time clock */
    clock_gettime( CLOCK_REALTIME, &ts );
  #endif

  if ( reference_seconds == 0.0 )
    reference_seconds = ( double ) ts.tv_sec;

  return ( ( double ) ts.tv_sec - reference_seconds )
         + ( double ) ts.tv_nsec * 1.0e-9;
}

#endif  /* __APPLE__ / __MACH__ */
#endif  /* _WIN32 */
