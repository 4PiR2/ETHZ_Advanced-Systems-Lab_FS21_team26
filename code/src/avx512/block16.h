#ifndef BLOCK16_H
#define BLOCK16_H

#include "immintrin.h"

inline void block16_transpose(__m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                              __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                              __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                              __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	__m512 ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8, ta9, ta10, ta11, ta12, ta13, ta14, ta15,
			tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tb10, tb11, tb12, tb13, tb14, tb15,
			tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8, tc9, tc10, tc11, tc12, tc13, tc14, tc15;
	__m512i idxb0 = _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0),
			idxb = _mm512_set1_epi32(4), idxb1 = idxb0 + idxb,
			idxc0 = _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0),
			idxc = idxb + idxb, idxc1 = idxc0 + idxc;
	ta0 = _mm512_unpacklo_ps(r0, r1);
	ta1 = _mm512_unpackhi_ps(r0, r1);
	ta2 = _mm512_unpacklo_ps(r2, r3);
	ta3 = _mm512_unpackhi_ps(r2, r3);
	ta4 = _mm512_unpacklo_ps(r4, r5);
	ta5 = _mm512_unpackhi_ps(r4, r5);
	ta6 = _mm512_unpacklo_ps(r6, r7);
	ta7 = _mm512_unpackhi_ps(r6, r7);
	ta8 = _mm512_unpacklo_ps(r8, r9);
	ta9 = _mm512_unpackhi_ps(r8, r9);
	ta10 = _mm512_unpacklo_ps(r10, r11);
	ta11 = _mm512_unpackhi_ps(r10, r11);
	ta12 = _mm512_unpacklo_ps(r12, r13);
	ta13 = _mm512_unpackhi_ps(r12, r13);
	ta14 = _mm512_unpacklo_ps(r14, r15);
	ta15 = _mm512_unpackhi_ps(r14, r15);
	tb0 = _mm512_shuffle_ps(ta0, ta2, _MM_SHUFFLE(1, 0, 1, 0));
	tb1 = _mm512_shuffle_ps(ta0, ta2, _MM_SHUFFLE(3, 2, 3, 2));
	tb2 = _mm512_shuffle_ps(ta1, ta3, _MM_SHUFFLE(1, 0, 1, 0));
	tb3 = _mm512_shuffle_ps(ta1, ta3, _MM_SHUFFLE(3, 2, 3, 2));
	tb4 = _mm512_shuffle_ps(ta4, ta6, _MM_SHUFFLE(1, 0, 1, 0));
	tb5 = _mm512_shuffle_ps(ta4, ta6, _MM_SHUFFLE(3, 2, 3, 2));
	tb6 = _mm512_shuffle_ps(ta5, ta7, _MM_SHUFFLE(1, 0, 1, 0));
	tb7 = _mm512_shuffle_ps(ta5, ta7, _MM_SHUFFLE(3, 2, 3, 2));
	tb8 = _mm512_shuffle_ps(ta8, ta10, _MM_SHUFFLE(1, 0, 1, 0));
	tb9 = _mm512_shuffle_ps(ta8, ta10, _MM_SHUFFLE(3, 2, 3, 2));
	tb10 = _mm512_shuffle_ps(ta9, ta11, _MM_SHUFFLE(1, 0, 1, 0));
	tb11 = _mm512_shuffle_ps(ta9, ta11, _MM_SHUFFLE(3, 2, 3, 2));
	tb12 = _mm512_shuffle_ps(ta12, ta14, _MM_SHUFFLE(1, 0, 1, 0));
	tb13 = _mm512_shuffle_ps(ta12, ta14, _MM_SHUFFLE(3, 2, 3, 2));
	tb14 = _mm512_shuffle_ps(ta13, ta15, _MM_SHUFFLE(1, 0, 1, 0));
	tb15 = _mm512_shuffle_ps(ta13, ta15, _MM_SHUFFLE(3, 2, 3, 2));
	tc0 = _mm512_permutex2var_ps(tb0, idxb0, tb4);
	tc1 = _mm512_permutex2var_ps(tb1, idxb0, tb5);
	tc2 = _mm512_permutex2var_ps(tb2, idxb0, tb6);
	tc3 = _mm512_permutex2var_ps(tb3, idxb0, tb7);
	tc4 = _mm512_permutex2var_ps(tb0, idxb1, tb4);
	tc5 = _mm512_permutex2var_ps(tb1, idxb1, tb5);
	tc6 = _mm512_permutex2var_ps(tb2, idxb1, tb6);
	tc7 = _mm512_permutex2var_ps(tb3, idxb1, tb7);
	tc8 = _mm512_permutex2var_ps(tb8, idxb0, tb12);
	tc9 = _mm512_permutex2var_ps(tb9, idxb0, tb13);
	tc10 = _mm512_permutex2var_ps(tb10, idxb0, tb14);
	tc11 = _mm512_permutex2var_ps(tb11, idxb0, tb15);
	tc12 = _mm512_permutex2var_ps(tb8, idxb1, tb12);
	tc13 = _mm512_permutex2var_ps(tb9, idxb1, tb13);
	tc14 = _mm512_permutex2var_ps(tb10, idxb1, tb14);
	tc15 = _mm512_permutex2var_ps(tb11, idxb1, tb15);
	r0 = _mm512_permutex2var_ps(tc0, idxc0, tc8);
	r1 = _mm512_permutex2var_ps(tc1, idxc0, tc9);
	r2 = _mm512_permutex2var_ps(tc2, idxc0, tc10);
	r3 = _mm512_permutex2var_ps(tc3, idxc0, tc11);
	r4 = _mm512_permutex2var_ps(tc4, idxc0, tc12);
	r5 = _mm512_permutex2var_ps(tc5, idxc0, tc13);
	r6 = _mm512_permutex2var_ps(tc6, idxc0, tc14);
	r7 = _mm512_permutex2var_ps(tc7, idxc0, tc15);
	r8 = _mm512_permutex2var_ps(tc0, idxc1, tc8);
	r9 = _mm512_permutex2var_ps(tc1, idxc1, tc9);
	r10 = _mm512_permutex2var_ps(tc2, idxc1, tc10);
	r11 = _mm512_permutex2var_ps(tc3, idxc1, tc11);
	r12 = _mm512_permutex2var_ps(tc4, idxc1, tc12);
	r13 = _mm512_permutex2var_ps(tc5, idxc1, tc13);
	r14 = _mm512_permutex2var_ps(tc6, idxc1, tc14);
	r15 = _mm512_permutex2var_ps(tc7, idxc1, tc15);
}

inline __m512 block16_row_sum(__m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                              __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                              __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                              __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	__m512 ta0, ta2, ta4, ta6, ta8, ta10, ta12, ta14, tb0, tb4, tb8, tb12, tc0, tc8;
	__m512i idxb0 = _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0),
			idxb = _mm512_set1_epi32(4), idxb1 = idxb0 + idxb,
			idxc0 = _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0),
			idxc = idxb + idxb, idxc1 = idxc0 + idxc;
	ta0 = _mm512_unpacklo_ps(r0, r1) + _mm512_unpackhi_ps(r0, r1);
	ta2 = _mm512_unpacklo_ps(r2, r3) + _mm512_unpackhi_ps(r2, r3);
	ta4 = _mm512_unpacklo_ps(r4, r5) + _mm512_unpackhi_ps(r4, r5);
	ta6 = _mm512_unpacklo_ps(r6, r7) + _mm512_unpackhi_ps(r6, r7);
	ta8 = _mm512_unpacklo_ps(r8, r9) + _mm512_unpackhi_ps(r8, r9);
	ta10 = _mm512_unpacklo_ps(r10, r11) + _mm512_unpackhi_ps(r10, r11);
	ta12 = _mm512_unpacklo_ps(r12, r13) + _mm512_unpackhi_ps(r12, r13);
	ta14 = _mm512_unpacklo_ps(r14, r15) + _mm512_unpackhi_ps(r14, r15);
	tb0 = _mm512_shuffle_ps(ta0, ta2, _MM_SHUFFLE(1, 0, 1, 0)) +
	      _mm512_shuffle_ps(ta0, ta2, _MM_SHUFFLE(3, 2, 3, 2));
	tb4 = _mm512_shuffle_ps(ta4, ta6, _MM_SHUFFLE(1, 0, 1, 0)) +
	      _mm512_shuffle_ps(ta4, ta6, _MM_SHUFFLE(3, 2, 3, 2));
	tb8 = _mm512_shuffle_ps(ta8, ta10, _MM_SHUFFLE(1, 0, 1, 0)) +
	      _mm512_shuffle_ps(ta8, ta10, _MM_SHUFFLE(3, 2, 3, 2));
	tb12 = _mm512_shuffle_ps(ta12, ta14, _MM_SHUFFLE(1, 0, 1, 0)) +
	       _mm512_shuffle_ps(ta12, ta14, _MM_SHUFFLE(3, 2, 3, 2));
	tc0 = _mm512_permutex2var_ps(tb0, idxb0, tb4) + _mm512_permutex2var_ps(tb0, idxb1, tb4);
	tc8 = _mm512_permutex2var_ps(tb8, idxb0, tb12) + _mm512_permutex2var_ps(tb8, idxb1, tb12);
	return _mm512_permutex2var_ps(tc0, idxc0, tc8) + _mm512_permutex2var_ps(tc0, idxc1, tc8);
}

inline void block16_load(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                         __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                         __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                         __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	r0 = _mm512_load_ps(p);
	r1 = _mm512_load_ps(p + M);
	r2 = _mm512_load_ps(p + M * 2);
	r3 = _mm512_load_ps(p + M * 3);
	r4 = _mm512_load_ps(p + M * 4);
	r5 = _mm512_load_ps(p + M * 5);
	r6 = _mm512_load_ps(p + M * 6);
	r7 = _mm512_load_ps(p + M * 7);
	r8 = _mm512_load_ps(p + M * 8);
	r9 = _mm512_load_ps(p + M * 9);
	r10 = _mm512_load_ps(p + M * 10);
	r11 = _mm512_load_ps(p + M * 11);
	r12 = _mm512_load_ps(p + M * 12);
	r13 = _mm512_load_ps(p + M * 13);
	r14 = _mm512_load_ps(p + M * 14);
	r15 = _mm512_load_ps(p + M * 15);
}

inline void block16_store(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                          __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                          __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                          __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	_mm512_store_ps(p, r0);
	_mm512_store_ps(p + M, r1);
	_mm512_store_ps(p + M * 2, r2);
	_mm512_store_ps(p + M * 3, r3);
	_mm512_store_ps(p + M * 4, r4);
	_mm512_store_ps(p + M * 5, r5);
	_mm512_store_ps(p + M * 6, r6);
	_mm512_store_ps(p + M * 7, r7);
	_mm512_store_ps(p + M * 8, r8);
	_mm512_store_ps(p + M * 9, r9);
	_mm512_store_ps(p + M * 10, r10);
	_mm512_store_ps(p + M * 11, r11);
	_mm512_store_ps(p + M * 12, r12);
	_mm512_store_ps(p + M * 13, r13);
	_mm512_store_ps(p + M * 14, r14);
	_mm512_store_ps(p + M * 15, r15);
}

void block16_debug_load(float **a, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                        __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                        __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                        __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	r0 = _mm512_loadu_ps(a[0]);
	r1 = _mm512_loadu_ps(a[1]);
	r2 = _mm512_loadu_ps(a[2]);
	r3 = _mm512_loadu_ps(a[3]);
	r4 = _mm512_loadu_ps(a[4]);
	r5 = _mm512_loadu_ps(a[5]);
	r6 = _mm512_loadu_ps(a[6]);
	r7 = _mm512_loadu_ps(a[7]);
	r8 = _mm512_loadu_ps(a[8]);
	r9 = _mm512_loadu_ps(a[9]);
	r10 = _mm512_loadu_ps(a[10]);
	r11 = _mm512_loadu_ps(a[11]);
	r12 = _mm512_loadu_ps(a[12]);
	r13 = _mm512_loadu_ps(a[13]);
	r14 = _mm512_loadu_ps(a[14]);
	r15 = _mm512_loadu_ps(a[15]);
}

void block16_debug_store(float **a, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                         __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                         __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                         __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	_mm512_storeu_ps(a[0], r0);
	_mm512_storeu_ps(a[1], r1);
	_mm512_storeu_ps(a[2], r2);
	_mm512_storeu_ps(a[3], r3);
	_mm512_storeu_ps(a[4], r4);
	_mm512_storeu_ps(a[5], r5);
	_mm512_storeu_ps(a[6], r6);
	_mm512_storeu_ps(a[7], r7);
	_mm512_storeu_ps(a[8], r8);
	_mm512_storeu_ps(a[9], r9);
	_mm512_storeu_ps(a[10], r10);
	_mm512_storeu_ps(a[11], r11);
	_mm512_storeu_ps(a[12], r12);
	_mm512_storeu_ps(a[13], r13);
	_mm512_storeu_ps(a[14], r14);
	_mm512_storeu_ps(a[15], r15);
}

#endif //BLOCK16_H
