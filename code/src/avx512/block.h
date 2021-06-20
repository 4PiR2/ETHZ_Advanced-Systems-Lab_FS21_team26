#ifndef BLOCK_H
#define BLOCK_H

#include "immintrin.h"

// 16x16
inline void block_load_transpose(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                                 __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                                 __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                                 __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	__m512i ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8, ta9, ta10, ta11, ta12, ta13, ta14, ta15,
			tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tb10, tb11, tb12, tb13, tb14, tb15,
			tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8, tc9, tc10, tc11, tc12, tc13, tc14, tc15,
			idx1 = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5),
			idx2 = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6),
			idx3 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	int mask;
	ta0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) p)),
	                         _mm256_load_si256((__m256i *) (p + M * 8)), 1);
	ta1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M))),
	                         _mm256_load_si256((__m256i *) (p + M * 9)), 1);
	ta2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 2))),
	                         _mm256_load_si256((__m256i *) (p + M * 10)), 1);
	ta3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 3))),
	                         _mm256_load_si256((__m256i *) (p + M * 11)), 1);
	ta4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 4))),
	                         _mm256_load_si256((__m256i *) (p + M * 12)), 1);
	ta5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 5))),
	                         _mm256_load_si256((__m256i *) (p + M * 13)), 1);
	ta6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 6))),
	                         _mm256_load_si256((__m256i *) (p + M * 14)), 1);
	ta7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 7))),
	                         _mm256_load_si256((__m256i *) (p + M * 15)), 1);
	ta8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + 8))),
	                         _mm256_load_si256((__m256i *) (p + M * 8 + 8)), 1);
	ta9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M + 8))),
	                         _mm256_load_si256((__m256i *) (p + M * 9 + 8)), 1);
	ta10 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 2 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 10 + 8)), 1);
	ta11 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 3 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 11 + 8)), 1);
	ta12 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 4 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 12 + 8)), 1);
	ta13 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 5 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 13 + 8)), 1);
	ta14 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 6 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 14 + 8)), 1);
	ta15 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i *) (p + M * 7 + 8))),
	                          _mm256_load_si256((__m256i *) (p + M * 15 + 8)), 1);
	mask = 0xcc;
	tb0 = _mm512_mask_permutexvar_epi64(ta0, (__mmask8) mask, idx1, ta4);
	tb1 = _mm512_mask_permutexvar_epi64(ta1, (__mmask8) mask, idx1, ta5);
	tb2 = _mm512_mask_permutexvar_epi64(ta2, (__mmask8) mask, idx1, ta6);
	tb3 = _mm512_mask_permutexvar_epi64(ta3, (__mmask8) mask, idx1, ta7);
	tb8 = _mm512_mask_permutexvar_epi64(ta8, (__mmask8) mask, idx1, ta12);
	tb9 = _mm512_mask_permutexvar_epi64(ta9, (__mmask8) mask, idx1, ta13);
	tb10 = _mm512_mask_permutexvar_epi64(ta10, (__mmask8) mask, idx1, ta14);
	tb11 = _mm512_mask_permutexvar_epi64(ta11, (__mmask8) mask, idx1, ta15);
	mask = 0x33;
	tb4 = _mm512_mask_permutexvar_epi64(ta4, (__mmask8) mask, idx1, ta0);
	tb5 = _mm512_mask_permutexvar_epi64(ta5, (__mmask8) mask, idx1, ta1);
	tb6 = _mm512_mask_permutexvar_epi64(ta6, (__mmask8) mask, idx1, ta2);
	tb7 = _mm512_mask_permutexvar_epi64(ta7, (__mmask8) mask, idx1, ta3);
	tb12 = _mm512_mask_permutexvar_epi64(ta12, (__mmask8) mask, idx1, ta8);
	tb13 = _mm512_mask_permutexvar_epi64(ta13, (__mmask8) mask, idx1, ta9);
	tb14 = _mm512_mask_permutexvar_epi64(ta14, (__mmask8) mask, idx1, ta10);
	tb15 = _mm512_mask_permutexvar_epi64(ta15, (__mmask8) mask, idx1, ta11);
	mask = 0xaa;
	tc0 = _mm512_mask_permutexvar_epi64(tb0, (__mmask8) mask, idx2, tb2);
	tc1 = _mm512_mask_permutexvar_epi64(tb1, (__mmask8) mask, idx2, tb3);
	tc4 = _mm512_mask_permutexvar_epi64(tb4, (__mmask8) mask, idx2, tb6);
	tc5 = _mm512_mask_permutexvar_epi64(tb5, (__mmask8) mask, idx2, tb7);
	tc8 = _mm512_mask_permutexvar_epi64(tb8, (__mmask8) mask, idx2, tb10);
	tc9 = _mm512_mask_permutexvar_epi64(tb9, (__mmask8) mask, idx2, tb11);
	tc12 = _mm512_mask_permutexvar_epi64(tb12, (__mmask8) mask, idx2, tb14);
	tc13 = _mm512_mask_permutexvar_epi64(tb13, (__mmask8) mask, idx2, tb15);
	mask = 0x55;
	tc2 = _mm512_mask_permutexvar_epi64(tb2, (__mmask8) mask, idx2, tb0);
	tc3 = _mm512_mask_permutexvar_epi64(tb3, (__mmask8) mask, idx2, tb1);
	tc6 = _mm512_mask_permutexvar_epi64(tb6, (__mmask8) mask, idx2, tb4);
	tc7 = _mm512_mask_permutexvar_epi64(tb7, (__mmask8) mask, idx2, tb5);
	tc10 = _mm512_mask_permutexvar_epi64(tb10, (__mmask8) mask, idx2, tb8);
	tc11 = _mm512_mask_permutexvar_epi64(tb11, (__mmask8) mask, idx2, tb9);
	tc14 = _mm512_mask_permutexvar_epi64(tb14, (__mmask8) mask, idx2, tb12);
	tc15 = _mm512_mask_permutexvar_epi64(tb15, (__mmask8) mask, idx2, tb13);
	mask = 0xaaaa;
	r0 = (__m512) _mm512_mask_permutexvar_epi32(tc0, (__mmask16) mask, idx3, tc1);
	r2 = (__m512) _mm512_mask_permutexvar_epi32(tc2, (__mmask16) mask, idx3, tc3);
	r4 = (__m512) _mm512_mask_permutexvar_epi32(tc4, (__mmask16) mask, idx3, tc5);
	r6 = (__m512) _mm512_mask_permutexvar_epi32(tc6, (__mmask16) mask, idx3, tc7);
	r8 = (__m512) _mm512_mask_permutexvar_epi32(tc8, (__mmask16) mask, idx3, tc9);
	r10 = (__m512) _mm512_mask_permutexvar_epi32(tc10, (__mmask16) mask, idx3, tc11);
	r12 = (__m512) _mm512_mask_permutexvar_epi32(tc12, (__mmask16) mask, idx3, tc13);
	r14 = (__m512) _mm512_mask_permutexvar_epi32(tc14, (__mmask16) mask, idx3, tc15);
	mask = 0x5555;
	r1 = (__m512) _mm512_mask_permutexvar_epi32(tc1, (__mmask16) mask, idx3, tc0);
	r3 = (__m512) _mm512_mask_permutexvar_epi32(tc3, (__mmask16) mask, idx3, tc2);
	r5 = (__m512) _mm512_mask_permutexvar_epi32(tc5, (__mmask16) mask, idx3, tc4);
	r7 = (__m512) _mm512_mask_permutexvar_epi32(tc7, (__mmask16) mask, idx3, tc6);
	r9 = (__m512) _mm512_mask_permutexvar_epi32(tc9, (__mmask16) mask, idx3, tc8);
	r11 = (__m512) _mm512_mask_permutexvar_epi32(tc11, (__mmask16) mask, idx3, tc10);
	r13 = (__m512) _mm512_mask_permutexvar_epi32(tc13, (__mmask16) mask, idx3, tc12);
	r15 = (__m512) _mm512_mask_permutexvar_epi32(tc15, (__mmask16) mask, idx3, tc14);
}

// 8x8
inline void block_load_transpose(float *p, int M, __m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
                                 __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
	__m256 ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7, v0, v2, v4, v6;
	ta0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p)), _mm_load_ps(p + M * 4), 1);
	ta1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M)), _mm_load_ps(p + M * 5), 1);
	ta2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M * 2)), _mm_load_ps(p + M * 6), 1);
	ta3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M * 3)), _mm_load_ps(p + M * 7), 1);
	ta4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + 4)), _mm_load_ps(p + M * 4 + 4), 1);
	ta5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M + 4)), _mm_load_ps(p + M * 5 + 4), 1);
	ta6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M * 2 + 4)), _mm_load_ps(p + M * 6 + 4), 1);
	ta7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p + M * 3 + 4)), _mm_load_ps(p + M * 7 + 4), 1);
	tb0 = _mm256_unpacklo_ps(ta0, ta1);
	tb1 = _mm256_unpackhi_ps(ta0, ta1);
	tb2 = _mm256_unpacklo_ps(ta2, ta3);
	tb3 = _mm256_unpackhi_ps(ta2, ta3);
	tb4 = _mm256_unpacklo_ps(ta4, ta5);
	tb5 = _mm256_unpackhi_ps(ta4, ta5);
	tb6 = _mm256_unpacklo_ps(ta6, ta7);
	tb7 = _mm256_unpackhi_ps(ta6, ta7);
	v0 = _mm256_shuffle_ps(tb0, tb2, 0x4e);
	r0 = _mm256_blend_ps(tb0, v0, 0xcc);
	r1 = _mm256_blend_ps(tb2, v0, 0x33);
	v2 = _mm256_shuffle_ps(tb1, tb3, 0x4e);
	r2 = _mm256_blend_ps(tb1, v2, 0xcc);
	r3 = _mm256_blend_ps(tb3, v2, 0x33);
	v4 = _mm256_shuffle_ps(tb4, tb6, 0x4e);
	r4 = _mm256_blend_ps(tb4, v4, 0xcc);
	r5 = _mm256_blend_ps(tb6, v4, 0x33);
	v6 = _mm256_shuffle_ps(tb5, tb7, 0x4e);
	r6 = _mm256_blend_ps(tb5, v6, 0xcc);
	r7 = _mm256_blend_ps(tb7, v6, 0x33);
}

// 16x16
inline void block_transpose(__m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                            __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                            __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                            __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	__m512 ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8, ta9, ta10, ta11, ta12, ta13, ta14, ta15,
			tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tb10, tb11, tb12, tb13, tb14, tb15,
			tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8, tc9, tc10, tc11, tc12, tc13, tc14, tc15;
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
	tb0 = _mm512_shuffle_ps(ta0, ta2, 0x44);
	tb1 = _mm512_shuffle_ps(ta0, ta2, 0xee);
	tb2 = _mm512_shuffle_ps(ta1, ta3, 0x44);
	tb3 = _mm512_shuffle_ps(ta1, ta3, 0xee);
	tb4 = _mm512_shuffle_ps(ta4, ta6, 0x44);
	tb5 = _mm512_shuffle_ps(ta4, ta6, 0xee);
	tb6 = _mm512_shuffle_ps(ta5, ta7, 0x44);
	tb7 = _mm512_shuffle_ps(ta5, ta7, 0xee);
	tb8 = _mm512_shuffle_ps(ta8, ta10, 0x44);
	tb9 = _mm512_shuffle_ps(ta8, ta10, 0xee);
	tb10 = _mm512_shuffle_ps(ta9, ta11, 0x44);
	tb11 = _mm512_shuffle_ps(ta9, ta11, 0xee);
	tb12 = _mm512_shuffle_ps(ta12, ta14, 0x44);
	tb13 = _mm512_shuffle_ps(ta12, ta14, 0xee);
	tb14 = _mm512_shuffle_ps(ta13, ta15, 0x44);
	tb15 = _mm512_shuffle_ps(ta13, ta15, 0xee);
	tc0 = _mm512_shuffle_f32x4(tb0, tb4, 0x88);
	tc1 = _mm512_shuffle_f32x4(tb1, tb5, 0x88);
	tc2 = _mm512_shuffle_f32x4(tb2, tb6, 0x88);
	tc3 = _mm512_shuffle_f32x4(tb3, tb7, 0x88);
	tc4 = _mm512_shuffle_f32x4(tb0, tb4, 0xdd);
	tc5 = _mm512_shuffle_f32x4(tb1, tb5, 0xdd);
	tc6 = _mm512_shuffle_f32x4(tb2, tb6, 0xdd);
	tc7 = _mm512_shuffle_f32x4(tb3, tb7, 0xdd);
	tc8 = _mm512_shuffle_f32x4(tb8, tb12, 0x88);
	tc9 = _mm512_shuffle_f32x4(tb9, tb13, 0x88);
	tc10 = _mm512_shuffle_f32x4(tb10, tb14, 0x88);
	tc11 = _mm512_shuffle_f32x4(tb11, tb15, 0x88);
	tc12 = _mm512_shuffle_f32x4(tb8, tb12, 0xdd);
	tc13 = _mm512_shuffle_f32x4(tb9, tb13, 0xdd);
	tc14 = _mm512_shuffle_f32x4(tb10, tb14, 0xdd);
	tc15 = _mm512_shuffle_f32x4(tb11, tb15, 0xdd);
	r0 = _mm512_shuffle_f32x4(tc0, tc8, 0x88);
	r1 = _mm512_shuffle_f32x4(tc1, tc9, 0x88);
	r2 = _mm512_shuffle_f32x4(tc2, tc10, 0x88);
	r3 = _mm512_shuffle_f32x4(tc3, tc11, 0x88);
	r4 = _mm512_shuffle_f32x4(tc4, tc12, 0x88);
	r5 = _mm512_shuffle_f32x4(tc5, tc13, 0x88);
	r6 = _mm512_shuffle_f32x4(tc6, tc14, 0x88);
	r7 = _mm512_shuffle_f32x4(tc7, tc15, 0x88);
	r8 = _mm512_shuffle_f32x4(tc0, tc8, 0xdd);
	r9 = _mm512_shuffle_f32x4(tc1, tc9, 0xdd);
	r10 = _mm512_shuffle_f32x4(tc2, tc10, 0xdd);
	r11 = _mm512_shuffle_f32x4(tc3, tc11, 0xdd);
	r12 = _mm512_shuffle_f32x4(tc4, tc12, 0xdd);
	r13 = _mm512_shuffle_f32x4(tc5, tc13, 0xdd);
	r14 = _mm512_shuffle_f32x4(tc6, tc14, 0xdd);
	r15 = _mm512_shuffle_f32x4(tc7, tc15, 0xdd);
}

// 16x16
inline __m512 block_row_sum(__m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                            __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7,
                            __m512 &r8, __m512 &r9, __m512 &r10, __m512 &r11,
                            __m512 &r12, __m512 &r13, __m512 &r14, __m512 &r15) {
	__m512 ta0, ta2, ta4, ta6, ta8, ta10, ta12, ta14, tb0, tb4, tb8, tb12, tc0, tc8;
	ta0 = _mm512_unpacklo_ps(r0, r1) + _mm512_unpackhi_ps(r0, r1);
	ta2 = _mm512_unpacklo_ps(r2, r3) + _mm512_unpackhi_ps(r2, r3);
	ta4 = _mm512_unpacklo_ps(r4, r5) + _mm512_unpackhi_ps(r4, r5);
	ta6 = _mm512_unpacklo_ps(r6, r7) + _mm512_unpackhi_ps(r6, r7);
	ta8 = _mm512_unpacklo_ps(r8, r9) + _mm512_unpackhi_ps(r8, r9);
	ta10 = _mm512_unpacklo_ps(r10, r11) + _mm512_unpackhi_ps(r10, r11);
	ta12 = _mm512_unpacklo_ps(r12, r13) + _mm512_unpackhi_ps(r12, r13);
	ta14 = _mm512_unpacklo_ps(r14, r15) + _mm512_unpackhi_ps(r14, r15);
	tb0 = _mm512_shuffle_ps(ta0, ta2, 0x44) + _mm512_shuffle_ps(ta0, ta2, 0xee);
	tb4 = _mm512_shuffle_ps(ta4, ta6, 0x44) + _mm512_shuffle_ps(ta4, ta6, 0xee);
	tb8 = _mm512_shuffle_ps(ta8, ta10, 0x44) + _mm512_shuffle_ps(ta8, ta10, 0xee);
	tb12 = _mm512_shuffle_ps(ta12, ta14, 0x44) + _mm512_shuffle_ps(ta12, ta14, 0xee);
	tc0 = _mm512_shuffle_f32x4(tb0, tb4, 0x88) + _mm512_shuffle_f32x4(tb0, tb4, 0xdd);
	tc8 = _mm512_shuffle_f32x4(tb8, tb12, 0x88) + _mm512_shuffle_f32x4(tb8, tb12, 0xdd);
	return _mm512_shuffle_f32x4(tc0, tc8, 0x88) + _mm512_shuffle_f32x4(tc0, tc8, 0xdd);
}

// 16x16
inline void block_load(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
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

// 8x16
inline void block_load(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
                       __m512 &r4, __m512 &r5, __m512 &r6, __m512 &r7) {
	r0 = _mm512_load_ps(p);
	r1 = _mm512_load_ps(p + M);
	r2 = _mm512_load_ps(p + M * 2);
	r3 = _mm512_load_ps(p + M * 3);
	r4 = _mm512_load_ps(p + M * 4);
	r5 = _mm512_load_ps(p + M * 5);
	r6 = _mm512_load_ps(p + M * 6);
	r7 = _mm512_load_ps(p + M * 7);
}

// 16x16
inline void block_store(float *p, int M, __m512 &r0, __m512 &r1, __m512 &r2, __m512 &r3,
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

// 8x8
inline void block_store(float *p, int M, __m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
                        __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
	_mm256_store_ps(p, r0);
	_mm256_store_ps(p + M, r1);
	_mm256_store_ps(p + M * 2, r2);
	_mm256_store_ps(p + M * 3, r3);
	_mm256_store_ps(p + M * 4, r4);
	_mm256_store_ps(p + M * 5, r5);
	_mm256_store_ps(p + M * 6, r6);
	_mm256_store_ps(p + M * 7, r7);
}

#endif //BLOCK_H
