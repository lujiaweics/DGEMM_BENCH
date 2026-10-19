#ifndef KERNEL_H_
#define KERNEL_H_

#include "common.h"
#include <immintrin.h>

#define PACKED

#ifndef PACKED
#if MR == 4
#define MR 4
#define NR 4

void Micro_kernel(int k, const double *A, int ldA, const double *B, int ldB, double *C, int ldC) {
    __m256d c_00_40 = _mm256_loadu_pd(&gamma(0, 0));
    __m256d c_01_41 = _mm256_loadu_pd(&gamma(0, 1));
    __m256d c_02_42 = _mm256_loadu_pd(&gamma(0, 2));
    __m256d c_03_43 = _mm256_loadu_pd(&gamma(0, 3));

    for (int p = 0; p < k; ++p) {
        __m256d a_00_40 = _mm256_loadu_pd(&alpha(0, p));
        __m256d b_p0 = _mm256_broadcast_sd(&beta(p, 0));
        c_00_40 = _mm256_fmadd_pd(a_00_40, b_p0, c_00_40);

        __m256d b_p1 = _mm256_broadcast_sd(&beta(p, 1));
        c_01_41 = _mm256_fmadd_pd(a_00_40, b_p1, c_01_41);

        __m256d b_p2 = _mm256_broadcast_sd(&beta(p, 2));
        c_02_42 = _mm256_fmadd_pd(a_00_40, b_p2, c_02_42);

        __m256d b_p3 = _mm256_broadcast_sd(&beta(p, 3));
        c_03_43 = _mm256_fmadd_pd(a_00_40, b_p3, c_03_43);
    }

    _mm256_storeu_pd(&gamma(0, 0), c_00_40);
    _mm256_storeu_pd(&gamma(0, 1), c_01_41);
    _mm256_storeu_pd(&gamma(0, 2), c_02_42);
    _mm256_storeu_pd(&gamma(0, 3), c_03_43);
}
#elif MR == 8
#define MR 8
#define NR 4

void Micro_kernel(int k, const double *A, int ldA, const double *B, int ldB, double *C, int ldC) {
    __m256d c_00_40 = _mm256_loadu_pd(&gamma(0, 0));
    __m256d c_40_80 = _mm256_loadu_pd(&gamma(4, 0));
    __m256d c_01_41 = _mm256_loadu_pd(&gamma(0, 1));
    __m256d c_41_81 = _mm256_loadu_pd(&gamma(4, 1));
    __m256d c_02_42 = _mm256_loadu_pd(&gamma(0, 2));
    __m256d c_42_82 = _mm256_loadu_pd(&gamma(4, 2));
    __m256d c_03_43 = _mm256_loadu_pd(&gamma(0, 3));
    __m256d c_43_83 = _mm256_loadu_pd(&gamma(4, 3));

    for (int p = 0; p < k; ++p) {
        __m256d a_00_40 = _mm256_loadu_pd(&alpha(0, p));
        __m256d a_40_80 = _mm256_loadu_pd(&alpha(4, p));
        __m256d b_p0 = _mm256_broadcast_sd(&beta(p, 0));
        c_00_40 = _mm256_fmadd_pd(a_00_40, b_p0, c_00_40);
        c_40_80 = _mm256_fmadd_pd(a_40_80, b_p0, c_40_80);

        __m256d b_p1 = _mm256_broadcast_sd(&beta(p, 1));
        c_01_41 = _mm256_fmadd_pd(a_00_40, b_p1, c_01_41);
        c_41_81 = _mm256_fmadd_pd(a_40_80, b_p1, c_41_81);

        __m256d b_p2 = _mm256_broadcast_sd(&beta(p, 2));
        c_02_42 = _mm256_fmadd_pd(a_00_40, b_p2, c_02_42);
        c_42_82 = _mm256_fmadd_pd(a_40_80, b_p2, c_42_82);

        __m256d b_p3 = _mm256_broadcast_sd(&beta(p, 3));
        c_03_43 = _mm256_fmadd_pd(a_00_40, b_p3, c_03_43);
        c_43_83 = _mm256_fmadd_pd(a_40_80, b_p3, c_43_83);
    }

    _mm256_storeu_pd(&gamma(0, 0), c_00_40);
    _mm256_storeu_pd(&gamma(4, 0), c_40_80);
    _mm256_storeu_pd(&gamma(0, 1), c_01_41);
    _mm256_storeu_pd(&gamma(4, 1), c_41_81);
    _mm256_storeu_pd(&gamma(0, 2), c_02_42);
    _mm256_storeu_pd(&gamma(4, 2), c_42_82);
    _mm256_storeu_pd(&gamma(0, 3), c_03_43);
    _mm256_storeu_pd(&gamma(4, 3), c_43_83);
}
#endif
#else
#if MR == 4
#define MR 4
#define NR 4

void Micro_kernel_Packed(int k, const double *packed_A, const double *packed_B, double *C, int ldC) {
    __m256d c_00_40 = _mm256_loadu_pd(&gamma(0, 0));
    __m256d c_01_41 = _mm256_loadu_pd(&gamma(0, 1));
    __m256d c_02_42 = _mm256_loadu_pd(&gamma(0, 2));
    __m256d c_03_43 = _mm256_loadu_pd(&gamma(0, 3));

    for (int p = 0; p < k; ++p) {
        __m256d a_00_40 = _mm256_loadu_pd(packed_A);
        __m256d b_p0 = _mm256_broadcast_sd(packed_B);
        c_00_40 = _mm256_fmadd_pd(a_00_40, b_p0, c_00_40);

        __m256d b_p1 = _mm256_broadcast_sd(packed_B + 1);
        c_01_41 = _mm256_fmadd_pd(a_00_40, b_p1, c_01_41);

        __m256d b_p2 = _mm256_broadcast_sd(packed_B + 2);
        c_02_42 = _mm256_fmadd_pd(a_00_40, b_p2, c_02_42);

        __m256d b_p3 = _mm256_broadcast_sd(packed_B + 3);
        c_03_43 = _mm256_fmadd_pd(a_00_40, b_p3, c_03_43);

        packed_A += MR;
        packed_B += NR;
    }

    _mm256_storeu_pd(&gamma(0, 0), c_00_40);
    _mm256_storeu_pd(&gamma(0, 1), c_01_41);
    _mm256_storeu_pd(&gamma(0, 2), c_02_42);
    _mm256_storeu_pd(&gamma(0, 3), c_03_43);
}
#elif MR == 8
#define MR 8
#define NR 4

void Micro_kernel_Packed(int k, const double *packed_A, const double *packed_B, double *C, int ldC) {
    __m256d c_00_40 = _mm256_loadu_pd(&gamma(0, 0));
    __m256d c_40_80 = _mm256_loadu_pd(&gamma(4, 0));
    __m256d c_01_41 = _mm256_loadu_pd(&gamma(0, 1));
    __m256d c_41_81 = _mm256_loadu_pd(&gamma(4, 1));
    __m256d c_02_42 = _mm256_loadu_pd(&gamma(0, 2));
    __m256d c_42_82 = _mm256_loadu_pd(&gamma(4, 2));
    __m256d c_03_43 = _mm256_loadu_pd(&gamma(0, 3));
    __m256d c_43_83 = _mm256_loadu_pd(&gamma(4, 3));

    for (int p = 0; p < k; ++p) {
        __m256d a_00_40 = _mm256_loadu_pd(packed_A);
        __m256d a_40_80 = _mm256_loadu_pd(packed_A + 4);
        __m256d b_p0 = _mm256_broadcast_sd(packed_B);
        c_00_40 = _mm256_fmadd_pd(a_00_40, b_p0, c_00_40);
        c_40_80 = _mm256_fmadd_pd(a_40_80, b_p0, c_40_80);

        __m256d b_p1 = _mm256_broadcast_sd(packed_B + 1);
        c_01_41 = _mm256_fmadd_pd(a_00_40, b_p1, c_01_41);
        c_41_81 = _mm256_fmadd_pd(a_40_80, b_p1, c_41_81);

        __m256d b_p2 = _mm256_broadcast_sd(packed_B + 2);
        c_02_42 = _mm256_fmadd_pd(a_00_40, b_p2, c_02_42);
        c_42_82 = _mm256_fmadd_pd(a_40_80, b_p2, c_42_82);

        __m256d b_p3 = _mm256_broadcast_sd(packed_B + 3);
        c_03_43 = _mm256_fmadd_pd(a_00_40, b_p3, c_03_43);
        c_43_83 = _mm256_fmadd_pd(a_40_80, b_p3, c_43_83);

        packed_A += MR;
        packed_B += NR;
    }

    _mm256_storeu_pd(&gamma(0, 0), c_00_40);
    _mm256_storeu_pd(&gamma(4, 0), c_40_80);
    _mm256_storeu_pd(&gamma(0, 1), c_01_41);
    _mm256_storeu_pd(&gamma(4, 1), c_41_81);
    _mm256_storeu_pd(&gamma(0, 2), c_02_42);
    _mm256_storeu_pd(&gamma(4, 2), c_42_82);
    _mm256_storeu_pd(&gamma(0, 3), c_03_43);
    _mm256_storeu_pd(&gamma(4, 3), c_43_83);
}
#endif

#endif

#endif