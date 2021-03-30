#ifndef CPP_LOW_DIM_H
#define CPP_LOW_DIM_H

#include "immintrin.h"
#include "block.h"

float low_dim_pair_aff(float *t, float *y, int n_samples, int d_out) {
	// t: lower 16-blocked triangle, diag elem = 1, margin = 0
	// output: sum of matrix without diag
	__m512i idx, ones = _mm512_set1_epi32(1);
	__m512 b, a, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, zerofs = _mm512_setzero_ps(),
			c = zerofs, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, csum, csub = zerofs,
			onefs = (__m512) _mm512_set1_epi32(0x3f800000);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0; i < N; i += 16) {
		for (int j = 0; j <= i; j += 16) {
			c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = zerofs;
			for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
				a = _mm512_load_ps(y + kN + i);
				b = _mm512_load_ps(y + kN + j);
				idx = _mm512_setzero_epi32();
				a0 = _mm512_permutexvar_ps(idx, a);
				a0 -= b;
				c0 = _mm512_fmadd_ps(a0, a0, c0);
				idx += ones;
				a1 = _mm512_permutexvar_ps(idx, a);
				a1 -= b;
				c1 = _mm512_fmadd_ps(a1, a1, c1);
				idx += ones;
				a2 = _mm512_permutexvar_ps(idx, a);
				a2 -= b;
				c2 = _mm512_fmadd_ps(a2, a2, c2);
				idx += ones;
				a3 = _mm512_permutexvar_ps(idx, a);
				a3 -= b;
				c3 = _mm512_fmadd_ps(a3, a3, c3);
				idx += ones;
				a4 = _mm512_permutexvar_ps(idx, a);
				a4 -= b;
				c4 = _mm512_fmadd_ps(a4, a4, c4);
				idx += ones;
				a5 = _mm512_permutexvar_ps(idx, a);
				a5 -= b;
				c5 = _mm512_fmadd_ps(a5, a5, c5);
				idx += ones;
				a6 = _mm512_permutexvar_ps(idx, a);
				a6 -= b;
				c6 = _mm512_fmadd_ps(a6, a6, c6);
				idx += ones;
				a7 = _mm512_permutexvar_ps(idx, a);
				a7 -= b;
				c7 = _mm512_fmadd_ps(a7, a7, c7);
				idx += ones;
				a8 = _mm512_permutexvar_ps(idx, a);
				a8 -= b;
				c8 = _mm512_fmadd_ps(a8, a8, c8);
				idx += ones;
				a9 = _mm512_permutexvar_ps(idx, a);
				a9 -= b;
				c9 = _mm512_fmadd_ps(a9, a9, c9);
				idx += ones;
				a10 = _mm512_permutexvar_ps(idx, a);
				a10 -= b;
				c10 = _mm512_fmadd_ps(a10, a10, c10);
				idx += ones;
				a11 = _mm512_permutexvar_ps(idx, a);
				a11 -= b;
				c11 = _mm512_fmadd_ps(a11, a11, c11);
				idx += ones;
				a12 = _mm512_permutexvar_ps(idx, a);
				a12 -= b;
				c12 = _mm512_fmadd_ps(a12, a12, c12);
				idx += ones;
				a13 = _mm512_permutexvar_ps(idx, a);
				a13 -= b;
				c13 = _mm512_fmadd_ps(a13, a13, c13);
				idx += ones;
				a14 = _mm512_permutexvar_ps(idx, a);
				a14 -= b;
				c14 = _mm512_fmadd_ps(a14, a14, c14);
				idx += ones;
				a15 = _mm512_permutexvar_ps(idx, a);
				a15 -= b;
				c15 = _mm512_fmadd_ps(a15, a15, c15);
			}
			float *s = t + i * N + j;
			c0 = _mm512_rcp14_ps(c0 + onefs);
			_mm512_store_ps(s, c0);
			s += N;
			c1 = _mm512_rcp14_ps(c1 + onefs);
			_mm512_store_ps(s, c1);
			s += N;
			c2 = _mm512_rcp14_ps(c2 + onefs);
			_mm512_store_ps(s, c2);
			s += N;
			c3 = _mm512_rcp14_ps(c3 + onefs);
			_mm512_store_ps(s, c3);
			s += N;
			c4 = _mm512_rcp14_ps(c4 + onefs);
			_mm512_store_ps(s, c4);
			s += N;
			c5 = _mm512_rcp14_ps(c5 + onefs);
			_mm512_store_ps(s, c5);
			s += N;
			c6 = _mm512_rcp14_ps(c6 + onefs);
			_mm512_store_ps(s, c6);
			s += N;
			c7 = _mm512_rcp14_ps(c7 + onefs);
			_mm512_store_ps(s, c7);
			s += N;
			c8 = _mm512_rcp14_ps(c8 + onefs);
			_mm512_store_ps(s, c8);
			s += N;
			c9 = _mm512_rcp14_ps(c9 + onefs);
			_mm512_store_ps(s, c9);
			s += N;
			c10 = _mm512_rcp14_ps(c10 + onefs);
			_mm512_store_ps(s, c10);
			s += N;
			c11 = _mm512_rcp14_ps(c11 + onefs);
			_mm512_store_ps(s, c11);
			s += N;
			c12 = _mm512_rcp14_ps(c12 + onefs);
			_mm512_store_ps(s, c12);
			s += N;
			c13 = _mm512_rcp14_ps(c13 + onefs);
			_mm512_store_ps(s, c13);
			s += N;
			c14 = _mm512_rcp14_ps(c14 + onefs);
			_mm512_store_ps(s, c14);
			s += N;
			c15 = _mm512_rcp14_ps(c15 + onefs);
			_mm512_store_ps(s, c15);
			csum = (((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7))) +
			       (((c8 + c9) + (c10 + c11)) + ((c12 + c13) + (c14 + c15)));
			c += csum;
			if (i != j) {
				c += csum;
			}
		}
	}
	for (int i = N - 16, iN = i * N; i < n_samples; ++i, iN += N) {
		for (int j = n_samples; j < N; ++j) {
			t[iN + j] = 0;
		}
	}
	for (int i = n_samples, iN = i * N; i < N; ++i, iN += N) {
		for (int j = 0; j < N; j += 16) {
			csub += _mm512_load_ps(t + iN + j);
			_mm512_store_ps(t + iN + j, zerofs);
		}
	}
	c -= csub + csub;
	return _mm512_reduce_add_ps(c) + (N - n_samples) * (N - n_samples) - n_samples;
}

void low_dim_update_calc(float *u, float *y, float *p, float *t, float t_sum, float eta, int n_samples, int d_out) {
	__m512i idx, ones = _mm512_set1_epi32(1);
	__m512 b, a, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, ui, uj, csum, rsum,
			p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
			t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
			t_rcp = _mm512_set1_ps(-1.f / t_sum), etax4 = _mm512_set1_ps(eta * 4.f);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0, iN = 0; i < N; i += 16, iN += N * 16) {
		for (int j = 0; j <= i; j += 16) {
			int iNj = i * N + j;
			p0 = _mm512_load_ps(p + iNj);
			t0 = _mm512_load_ps(t + iNj);
			t0 = _mm512_fmadd_ps(t0, t_rcp, p0) * t0;
			iNj += N;
			p1 = _mm512_load_ps(p + iNj);
			t1 = _mm512_load_ps(t + iNj);
			t1 = _mm512_fmadd_ps(t1, t_rcp, p1) * t1;
			iNj += N;
			p2 = _mm512_load_ps(p + iNj);
			t2 = _mm512_load_ps(t + iNj);
			t2 = _mm512_fmadd_ps(t2, t_rcp, p2) * t2;
			iNj += N;
			p3 = _mm512_load_ps(p + iNj);
			t3 = _mm512_load_ps(t + iNj);
			t3 = _mm512_fmadd_ps(t3, t_rcp, p3) * t3;
			iNj += N;
			p4 = _mm512_load_ps(p + iNj);
			t4 = _mm512_load_ps(t + iNj);
			t4 = _mm512_fmadd_ps(t4, t_rcp, p4) * t4;
			iNj += N;
			p5 = _mm512_load_ps(p + iNj);
			t5 = _mm512_load_ps(t + iNj);
			t5 = _mm512_fmadd_ps(t5, t_rcp, p5) * t5;
			iNj += N;
			p6 = _mm512_load_ps(p + iNj);
			t6 = _mm512_load_ps(t + iNj);
			t6 = _mm512_fmadd_ps(t6, t_rcp, p6) * t6;
			iNj += N;
			p7 = _mm512_load_ps(p + iNj);
			t7 = _mm512_load_ps(t + iNj);
			t7 = _mm512_fmadd_ps(t7, t_rcp, p7) * t7;
			iNj += N;
			p8 = _mm512_load_ps(p + iNj);
			t8 = _mm512_load_ps(t + iNj);
			t8 = _mm512_fmadd_ps(t8, t_rcp, p8) * t8;
			iNj += N;
			p9 = _mm512_load_ps(p + iNj);
			t9 = _mm512_load_ps(t + iNj);
			t9 = _mm512_fmadd_ps(t9, t_rcp, p9) * t9;
			iNj += N;
			p10 = _mm512_load_ps(p + iNj);
			t10 = _mm512_load_ps(t + iNj);
			t10 = _mm512_fmadd_ps(t10, t_rcp, p10) * t10;
			iNj += N;
			p11 = _mm512_load_ps(p + iNj);
			t11 = _mm512_load_ps(t + iNj);
			t11 = _mm512_fmadd_ps(t11, t_rcp, p11) * t11;
			iNj += N;
			p12 = _mm512_load_ps(p + iNj);
			t12 = _mm512_load_ps(t + iNj);
			t12 = _mm512_fmadd_ps(t12, t_rcp, p12) * t12;
			iNj += N;
			p13 = _mm512_load_ps(p + iNj);
			t13 = _mm512_load_ps(t + iNj);
			t13 = _mm512_fmadd_ps(t13, t_rcp, p13) * t13;
			iNj += N;
			p14 = _mm512_load_ps(p + iNj);
			t14 = _mm512_load_ps(t + iNj);
			t14 = _mm512_fmadd_ps(t14, t_rcp, p14) * t14;
			iNj += N;
			p15 = _mm512_load_ps(p + iNj);
			t15 = _mm512_load_ps(t + iNj);
			t15 = _mm512_fmadd_ps(t15, t_rcp, p15) * t15;
			for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
				a = _mm512_load_ps(y + kN + i);
				b = _mm512_load_ps(y + kN + j);
				idx = _mm512_setzero_epi32();
				a0 = _mm512_permutexvar_ps(idx, a);
				a0 -= b;
				a0 *= t0;
				idx += ones;
				a1 = _mm512_permutexvar_ps(idx, a);
				a1 -= b;
				a1 *= t1;
				idx += ones;
				a2 = _mm512_permutexvar_ps(idx, a);
				a2 -= b;
				a2 *= t2;
				idx += ones;
				a3 = _mm512_permutexvar_ps(idx, a);
				a3 -= b;
				a3 *= t3;
				idx += ones;
				a4 = _mm512_permutexvar_ps(idx, a);
				a4 -= b;
				a4 *= t4;
				idx += ones;
				a5 = _mm512_permutexvar_ps(idx, a);
				a5 -= b;
				a5 *= t5;
				idx += ones;
				a6 = _mm512_permutexvar_ps(idx, a);
				a6 -= b;
				a6 *= t6;
				idx += ones;
				a7 = _mm512_permutexvar_ps(idx, a);
				a7 -= b;
				a7 *= t7;
				idx += ones;
				a8 = _mm512_permutexvar_ps(idx, a);
				a8 -= b;
				a8 *= t8;
				idx += ones;
				a9 = _mm512_permutexvar_ps(idx, a);
				a9 -= b;
				a9 *= t9;
				idx += ones;
				a10 = _mm512_permutexvar_ps(idx, a);
				a10 -= b;
				a10 *= t10;
				idx += ones;
				a11 = _mm512_permutexvar_ps(idx, a);
				a11 -= b;
				a11 *= t11;
				idx += ones;
				a12 = _mm512_permutexvar_ps(idx, a);
				a12 -= b;
				a12 *= t12;
				idx += ones;
				a13 = _mm512_permutexvar_ps(idx, a);
				a13 -= b;
				a13 *= t13;
				idx += ones;
				a14 = _mm512_permutexvar_ps(idx, a);
				a14 -= b;
				a14 *= t14;
				idx += ones;
				a15 = _mm512_permutexvar_ps(idx, a);
				a15 -= b;
				a15 *= t15;
				uj = _mm512_load_ps(u + kN + j);
				csum = (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
				       (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)));
				uj = _mm512_fnmadd_ps(csum, etax4, uj);
				_mm512_store_ps(u + kN + j, uj);
				if (i != j) {
					ui = _mm512_load_ps(u + kN + i);
					rsum = block_row_sum(a0, a1, a2, a3, a4, a5, a6, a7,
					                     a8, a9, a10, a11, a12, a13, a14, a15);
					ui = _mm512_fmadd_ps(rsum, etax4, ui);
					_mm512_store_ps(u + kN + i, ui);
				}
			}
		}
	}
}

void low_dim_update_apply(float *y, float *u, float alpha, int n_samples, int d_out) {
	int N = (n_samples + 15) & (-1 ^ 15);
	__m512 yi, ui, alphas = _mm512_set1_ps(alpha);
	for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
		for (int i = 0; i < N; i += 16) {
			yi = _mm512_load_ps(y + kN + i);
			ui = _mm512_load_ps(u + kN + i);
			yi += ui;
			ui *= alphas;
			_mm512_store_ps(y + kN + i, yi);
			_mm512_store_ps(u + kN + i, ui);
		}
	}
}

void low_dim_center(float *y, int n_samples, int d_out) {
	int N = (n_samples + 15) & (-1 ^ 15), lim = N - 48;
	__m512 y0, y1, y2, y3, s0, s1, s2, s3, ym;
	for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
		s0 = s1 = s2 = s3 = (__m512) _mm512_set1_epi32(0);
		int i;
		for (i = 0; i < lim; i += 64) {
			y0 = _mm512_load_ps(y + kN + i);
			y1 = _mm512_load_ps(y + kN + i + 16);
			y2 = _mm512_load_ps(y + kN + i + 32);
			y3 = _mm512_load_ps(y + kN + i + 48);
			s0 += y0;
			s1 += y1;
			s2 += y2;
			s3 += y3;
		}
		for (; i < N; i += 16) {
			y0 = _mm512_load_ps(y + kN + i);
			s0 += y0;
		}
		float sum = _mm512_reduce_add_ps((s0 + s1) + (s2 + s3));
		ym = _mm512_set1_ps(sum / n_samples);
		for (i = 0; i < N; i += 16) {
			y0 = _mm512_load_ps(y + kN + i);
			y0 -= ym;
			_mm512_store_ps(y + kN + i, y0);
		}
	}
}

#endif //CPP_LOW_DIM_H
