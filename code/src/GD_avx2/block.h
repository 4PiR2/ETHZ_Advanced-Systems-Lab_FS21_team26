#ifndef BLOCK_H
#define BLOCK_H

#include "immintrin.h"

// 8x8
inline void block_transpose(__m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
							__m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
	__m256 ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7;
	ta0 = _mm256_unpacklo_ps(r0, r1);
	ta1 = _mm256_unpackhi_ps(r0, r1);
	ta2 = _mm256_unpacklo_ps(r2, r3);
	ta3 = _mm256_unpackhi_ps(r2, r3);
	ta4 = _mm256_unpacklo_ps(r4, r5);
	ta5 = _mm256_unpackhi_ps(r4, r5);
	ta6 = _mm256_unpacklo_ps(r6, r7);
	ta7 = _mm256_unpackhi_ps(r6, r7);
	tb0 = _mm256_shuffle_ps(ta0, ta2, _MM_SHUFFLE(1, 0, 1, 0));
	tb1 = _mm256_shuffle_ps(ta0, ta2, _MM_SHUFFLE(3, 2, 3, 2));
	tb2 = _mm256_shuffle_ps(ta1, ta3, _MM_SHUFFLE(1, 0, 1, 0));
	tb3 = _mm256_shuffle_ps(ta1, ta3, _MM_SHUFFLE(3, 2, 3, 2));
	tb4 = _mm256_shuffle_ps(ta4, ta6, _MM_SHUFFLE(1, 0, 1, 0));
	tb5 = _mm256_shuffle_ps(ta4, ta6, _MM_SHUFFLE(3, 2, 3, 2));
	tb6 = _mm256_shuffle_ps(ta5, ta7, _MM_SHUFFLE(1, 0, 1, 0));
	tb7 = _mm256_shuffle_ps(ta5, ta7, _MM_SHUFFLE(3, 2, 3, 2));
	r0 = _mm256_permute2f128_ps(tb0, tb4, 0x20);
	r1 = _mm256_permute2f128_ps(tb1, tb5, 0x20);
	r2 = _mm256_permute2f128_ps(tb2, tb6, 0x20);
	r3 = _mm256_permute2f128_ps(tb3, tb7, 0x20);
	r4 = _mm256_permute2f128_ps(tb0, tb4, 0x31);
	r5 = _mm256_permute2f128_ps(tb1, tb5, 0x31);
	r6 = _mm256_permute2f128_ps(tb2, tb6, 0x31);
	r7 = _mm256_permute2f128_ps(tb3, tb7, 0x31);
}

// 8x8
inline __m256 block_row_sum(__m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
                            __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
	__m256 ta0, ta2, ta4, ta6, tb0, tb4, tc0;
	//__m256i idxb0 = _mm256_set_epi32(11, 10, 9, 8, 3, 2, 1, 0),
	//		idxb = _mm256_set1_epi32(4), idxb1 = idxb0 + idxb;
	ta0 = _mm256_unpacklo_ps(r0, r1) + _mm256_unpackhi_ps(r0, r1);
	ta2 = _mm256_unpacklo_ps(r2, r3) + _mm256_unpackhi_ps(r2, r3);
	ta4 = _mm256_unpacklo_ps(r4, r5) + _mm256_unpackhi_ps(r4, r5);
	ta6 = _mm256_unpacklo_ps(r6, r7) + _mm256_unpackhi_ps(r6, r7);
	tb0 = _mm256_shuffle_ps(ta0, ta2, _MM_SHUFFLE(1, 0, 1, 0)) +
	      _mm256_shuffle_ps(ta0, ta2, _MM_SHUFFLE(3, 2, 3, 2));
	tb4 = _mm256_shuffle_ps(ta4, ta6, _MM_SHUFFLE(1, 0, 1, 0)) +
	      _mm256_shuffle_ps(ta4, ta6, _MM_SHUFFLE(3, 2, 3, 2));
	//tc0 = _mm256_permutex2var_ps(tb0, idxb0, tb4) + _mm256_permutex2var_ps(tb0, idxb1, tb4);
	tc0 = _mm256_permute2f128_ps(tb0, tb4, 0b00100000) + 
	      _mm256_permute2f128_ps(tb0, tb4, 0b00110001);
	return tc0;
}

// 8x8
inline void block_load(float *p, int M, __m256 &r0, __m256 &r1, __m256 &r2, __m256 &r3,
                       __m256 &r4, __m256 &r5, __m256 &r6, __m256 &r7) {
	r0 = _mm256_load_ps(p);
	r1 = _mm256_load_ps(p + M);
	r2 = _mm256_load_ps(p + M * 2);
	r3 = _mm256_load_ps(p + M * 3);
	r4 = _mm256_load_ps(p + M * 4);
	r5 = _mm256_load_ps(p + M * 5);
	r6 = _mm256_load_ps(p + M * 6);
	r7 = _mm256_load_ps(p + M * 7);
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
