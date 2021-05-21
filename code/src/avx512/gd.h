#ifndef GD_H
#define GD_H

#include "block16.h"

float gd_pair_aff(float *t, float *y, int n_samples, int d_out) {
	// t: lower 16-blocked triangle, diag elem = 1, margin = 0
	// output: sum of matrix without diag
	__m512i idx, ones = _mm512_set1_epi32(1);
	__m512 b, a, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, zerofs = _mm512_setzero_ps(),
			c = zerofs, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, csum, csub0 = zerofs,
			csub1 = zerofs, csub2 = zerofs, csub3 = zerofs,
			onefs = (__m512) _mm512_set1_epi32(0x3f800000);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0; i < N; i += 16) {
		for (int j = 0; j <= i; j += 16) {
			c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = onefs;
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
			c0 = _mm512_rcp14_ps(c0);
			c1 = _mm512_rcp14_ps(c1);
			c2 = _mm512_rcp14_ps(c2);
			c3 = _mm512_rcp14_ps(c3);
			c4 = _mm512_rcp14_ps(c4);
			c5 = _mm512_rcp14_ps(c5);
			c6 = _mm512_rcp14_ps(c6);
			c7 = _mm512_rcp14_ps(c7);
			c8 = _mm512_rcp14_ps(c8);
			c9 = _mm512_rcp14_ps(c9);
			c10 = _mm512_rcp14_ps(c10);
			c11 = _mm512_rcp14_ps(c11);
			c12 = _mm512_rcp14_ps(c12);
			c13 = _mm512_rcp14_ps(c13);
			c14 = _mm512_rcp14_ps(c14);
			c15 = _mm512_rcp14_ps(c15);
			block16_store(t + i * N + j, N, c0, c1, c2, c3, c4, c5, c6, c7,
			            c8, c9, c10, c11, c12, c13, c14, c15);
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
	int limit = N * N, lim = limit - 16 * (4 - 1), offset = n_samples * N;
	for (; offset < lim; offset += 16 * 4) {
		csub0 += _mm512_load_ps(t + offset);
		_mm512_store_ps(t + offset, zerofs);
		csub1 += _mm512_load_ps(t + offset + 16);
		_mm512_store_ps(t + offset + 16, zerofs);
		csub2 += _mm512_load_ps(t + offset + 32);
		_mm512_store_ps(t + offset + 32, zerofs);
		csub3 += _mm512_load_ps(t + offset + 48);
		_mm512_store_ps(t + offset + 48, zerofs);
	}
	for (; offset < limit; offset += 16) {
		csub0 += _mm512_load_ps(t + offset);
		_mm512_store_ps(t + offset, zerofs);
	}
	csub0 = (csub0 + csub1) + (csub2 + csub3);
	c = _mm512_fnmadd_ps(csub0, onefs + onefs, c);
	return _mm512_reduce_add_ps(c) + (float) ((N - n_samples) * (N - n_samples) - n_samples);
}

void gd_update_calc(float *u, float *y, float *p, float *t, float t_sum, float eta, int n_samples, int d_out) {
	// p: lower 16-blocked triangle
	__m512i idx, ones = _mm512_set1_epi32(1);
	__m512 b, a, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, ui, uj, csum, rsum,
			p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
			t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15,
			t_rcp = _mm512_set1_ps(1.f / t_sum), etax4 = _mm512_set1_ps(eta * 4.f);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0, iN = 0; i < N; i += 16, iN += N * 16) {
		for (int j = 0; j <= i; j += 16) {
			block16_load(p + i * N + j, N, p0, p1, p2, p3, p4, p5, p6, p7,
			           p8, p9, p10, p11, p12, p13, p14, p15);
			block16_load(t + i * N + j, N, t0, t1, t2, t3, t4, t5, t6, t7,
			           t8, t9, t10, t11, t12, t13, t14, t15);
			t0 = _mm512_fnmadd_ps(t0, t_rcp, p0) * t0;
			t1 = _mm512_fnmadd_ps(t1, t_rcp, p1) * t1;
			t2 = _mm512_fnmadd_ps(t2, t_rcp, p2) * t2;
			t3 = _mm512_fnmadd_ps(t3, t_rcp, p3) * t3;
			t4 = _mm512_fnmadd_ps(t4, t_rcp, p4) * t4;
			t5 = _mm512_fnmadd_ps(t5, t_rcp, p5) * t5;
			t6 = _mm512_fnmadd_ps(t6, t_rcp, p6) * t6;
			t7 = _mm512_fnmadd_ps(t7, t_rcp, p7) * t7;
			t8 = _mm512_fnmadd_ps(t8, t_rcp, p8) * t8;
			t9 = _mm512_fnmadd_ps(t9, t_rcp, p9) * t9;
			t10 = _mm512_fnmadd_ps(t10, t_rcp, p10) * t10;
			t11 = _mm512_fnmadd_ps(t11, t_rcp, p11) * t11;
			t12 = _mm512_fnmadd_ps(t12, t_rcp, p12) * t12;
			t13 = _mm512_fnmadd_ps(t13, t_rcp, p13) * t13;
			t14 = _mm512_fnmadd_ps(t14, t_rcp, p14) * t14;
			t15 = _mm512_fnmadd_ps(t15, t_rcp, p15) * t15;
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
				uj = _mm512_fmadd_ps(csum, etax4, uj);
				_mm512_store_ps(u + kN + j, uj);
				if (i != j) {
					ui = _mm512_load_ps(u + kN + i);
					rsum = block16_row_sum(a0, a1, a2, a3, a4, a5, a6, a7,
					                     a8, a9, a10, a11, a12, a13, a14, a15);
					ui = _mm512_fnmadd_ps(rsum, etax4, ui);
					_mm512_store_ps(u + kN + i, ui);
				}
			}
		}
	}
}

void gd_update_apply(float *y, float *u, float alpha, int n_samples, int d_out) {
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

#endif //GD_H
