#ifndef EXP_H
#define EXP_H

#include "immintrin.h"

inline __m512 exp_ps(__m512 x) {
	__m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
	__m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);

	__m512 cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);
	__m512 inv_LOG2EF = _mm512_set1_ps(.693147180559945f);

	__m512 cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
	__m512 cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
	__m512 cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
	__m512 cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
	__m512 cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
	__m512 cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);
	__m512 fx;
	__m512i imm0;
	__m512 one = _mm512_set1_ps(1.f);

	x = _mm512_min_ps(x, exp_hi);
	x = _mm512_max_ps(x, exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = x * cephes_LOG2EF;
	fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT);
	x = _mm512_fnmadd_ps(fx, inv_LOG2EF, x);

	__m512 y = cephes_exp_p0;
	y = _mm512_fmadd_ps(y, x, cephes_exp_p1);
	y = _mm512_fmadd_ps(y, x, cephes_exp_p2);
	y = _mm512_fmadd_ps(y, x, cephes_exp_p3);
	y = _mm512_fmadd_ps(y, x, cephes_exp_p4);
	y = _mm512_fmadd_ps(y, x, cephes_exp_p5);
	y = _mm512_fmadd_ps(y, x * x, x + one);

	/* build 2^n */
	imm0 = _mm512_cvttps_epi32(fx);
	imm0 = _mm512_add_epi32(imm0, _mm512_set1_epi32(0x7f));
	imm0 = _mm512_slli_epi32(imm0, 23);
	y *= (__m512) imm0;
	return y;
}

inline __m512 exp_app_ps(__m512 x) {
	// -1 <= x <= 0
	__m512 y,
			a0 = _mm512_set1_ps(.9996136409397813f),
			a1 = _mm512_set1_ps(.9920487460431511f),
			a2 = _mm512_set1_ps(.4624692123106021f),
			a3 = _mm512_set1_ps(.10250045262707179f);
	y = _mm512_fmadd_ps(a3, x, a2);
	y = _mm512_fmadd_ps(y, x, a1);
	y = _mm512_fmadd_ps(y, x, a0);
	return y;
}

#endif //EXP_H
