#ifndef GD_H
#define GD_H

#include "block.h"

float gd_pair_aff(float *t, float *y, int n_samples, int d_out) {
	// t: lower 8-blocked triangle, diag elem = 1, margin = 0
	// output: sum of matrix without diag
	__m256i idx, ones = _mm256_set1_epi32(1);
	__m256  zerofs = _mm256_setzero_ps(), onefs = (__m256) _mm256_set1_epi32(0x3f800000), //float 1.0
	a, a0, a1, a2, a3, a4, a5, a6, a7, b,
			c = zerofs, c0, c1, c2, c3, c4, c5, c6, c7, csum, csub0 = zerofs,
			csub1 = zerofs, csub2 = zerofs, csub3 = zerofs;
	int N = (n_samples + 7) & (-1 ^ 7);
	for (int i = 0; i < N; i += 8) {
		for (int j = 0; j <= i; j += 8) {
			c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = onefs;
			for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
				a = _mm256_load_ps(y + kN + i);
				b = _mm256_load_ps(y + kN + j);
				idx = _mm256_setzero_si256();
				a0 = _mm256_permutexvar_ps(idx, a);
				a0 -= b;
				c0 = _mm256_fmadd_ps(a0, a0, c0);
				idx += ones;
				a1 = _mm256_permutexvar_ps(idx, a);
				a1 -= b;
				c1 = _mm256_fmadd_ps(a1, a1, c1);
				idx += ones;
				a2 = _mm256_permutexvar_ps(idx, a);
				a2 -= b;
				c2 = _mm256_fmadd_ps(a2, a2, c2);
				idx += ones;
				a3 = _mm256_permutexvar_ps(idx, a);
				a3 -= b;
				c3 = _mm256_fmadd_ps(a3, a3, c3);
				idx += ones;
				a4 = _mm256_permutexvar_ps(idx, a);
				a4 -= b;
				c4 = _mm256_fmadd_ps(a4, a4, c4);
				idx += ones;
				a5 = _mm256_permutexvar_ps(idx, a);
				a5 -= b;
				c5 = _mm256_fmadd_ps(a5, a5, c5);
				idx += ones;
				a6 = _mm256_permutexvar_ps(idx, a);
				a6 -= b;
				c6 = _mm256_fmadd_ps(a6, a6, c6);
				idx += ones;
				a7 = _mm256_permutexvar_ps(idx, a);
				a7 -= b;
				c7 = _mm256_fmadd_ps(a7, a7, c7);
			}
			c0 = _mm256_rcp_ps(c0);
			c1 = _mm256_rcp_ps(c1);
			c2 = _mm256_rcp_ps(c2);
			c3 = _mm256_rcp_ps(c3);
			c4 = _mm256_rcp_ps(c4);
			c5 = _mm256_rcp_ps(c5);
			c6 = _mm256_rcp_ps(c6);
			c7 = _mm256_rcp_ps(c7);
			block_store(t + i * N + j, N, c0, c1, c2, c3, c4, c5, c6, c7);
			csum = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
			c += csum;
			if (i != j) {
				c += csum;
			} //sum of whole matrix
		}
	}
	for (int i = N - 8, iN = i * N; i < n_samples; ++i, iN += N) {
		for (int j = n_samples; j < N; ++j) {
			t[iN + j] = 0;
		}
	}
	int limit = N * N, lim = limit - 8 * (4 - 1), offset = n_samples * N;
	for (; offset < lim; offset += 8 * 4) {
		csub0 += _mm256_load_ps(t + offset);
		_mm256_store_ps(t + offset, zerofs);
		csub1 += _mm256_load_ps(t + offset + 8);
		_mm256_store_ps(t + offset + 8, zerofs);
		csub2 += _mm256_load_ps(t + offset + 16);
		_mm256_store_ps(t + offset + 16, zerofs);
		csub3 += _mm256_load_ps(t + offset + 24);
		_mm256_store_ps(t + offset + 24, zerofs);
	}
	for (; offset < limit; offset += 8) {
		csub0 += _mm256_load_ps(t + offset);
		_mm256_store_ps(t + offset, zerofs);
	}
	csub0 = (csub0 + csub1) + (csub2 + csub3);
	c = _mm256_fnmadd_ps(csub0, onefs + onefs, c);

	return ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7])) + (float) ((N - n_samples) * (N - n_samples) - n_samples);//no such function
}

void gd_update_calc(float *u, float *y, float *p, float *t, float t_sum, float eta, int n_samples, int d_out) {
	// p: lower 16-blocked triangle
	__m256i idx, ones = _mm256_set1_epi32(1);
	__m256  a, a0, a1, a2, a3, a4, a5, a6, a7,
			p0, p1, p2, p3, p4, p5, p6, p7,
			t0, t1, t2, t3, t4, t5, t6, t7,
			b, ui, uj, csum, rsum, t_rcp = _mm256_set1_ps(1.f / t_sum), etax4 = _mm256_set1_ps(eta * 4.f);
	int N = (n_samples + 7) & (-1 ^ 7);
	for (int i = 0, iN = 0; i < N; i += 8, iN += N * 8) {
		for (int j = 0; j <= i; j += 8) {
			block_load(p + i * N + j, N, p0, p1, p2, p3, p4, p5, p6, p7);
			block_load(t + i * N + j, N, t0, t1, t2, t3, t4, t5, t6, t7);
			t0 = _mm256_fnmadd_ps(t0, t_rcp, p0) * t0;
			t1 = _mm256_fnmadd_ps(t1, t_rcp, p1) * t1;
			t2 = _mm256_fnmadd_ps(t2, t_rcp, p2) * t2;
			t3 = _mm256_fnmadd_ps(t3, t_rcp, p3) * t3;
			t4 = _mm256_fnmadd_ps(t4, t_rcp, p4) * t4;
			t5 = _mm256_fnmadd_ps(t5, t_rcp, p5) * t5;
			t6 = _mm256_fnmadd_ps(t6, t_rcp, p6) * t6;
			t7 = _mm256_fnmadd_ps(t7, t_rcp, p7) * t7;
			for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
				a = _mm256_load_ps(y + kN + i);
				b = _mm256_load_ps(y + kN + j);
				idx = _mm256_setzero_si256();
				a0 = _mm256_permutexvar_ps(idx, a);
				a0 -= b;
				a0 *= t0;
				idx += ones;
				a1 = _mm256_permutexvar_ps(idx, a);
				a1 -= b;
				a1 *= t1;
				idx += ones;
				a2 = _mm256_permutexvar_ps(idx, a);
				a2 -= b;
				a2 *= t2;
				idx += ones;
				a3 = _mm256_permutexvar_ps(idx, a);
				a3 -= b;
				a3 *= t3;
				idx += ones;
				a4 = _mm256_permutexvar_ps(idx, a);
				a4 -= b;
				a4 *= t4;
				idx += ones;
				a5 = _mm256_permutexvar_ps(idx, a);
				a5 -= b;
				a5 *= t5;
				idx += ones;
				a6 = _mm256_permutexvar_ps(idx, a);
				a6 -= b;
				a6 *= t6;
				idx += ones;
				a7 = _mm256_permutexvar_ps(idx, a);
				a7 -= b;
				a7 *= t7;
				uj = _mm256_load_ps(u + kN + j);
				csum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
				uj = _mm256_fmadd_ps(csum, etax4, uj);
				_mm256_store_ps(u + kN + j, uj);
				if (i != j) {
					ui = _mm256_load_ps(u + kN + i);
					rsum = block_row_sum(a0, a1, a2, a3, a4, a5, a6, a7);//modify
					ui = _mm256_fnmadd_ps(rsum, etax4, ui);
					_mm256_store_ps(u + kN + i, ui);
				}
			}
		}
	}
}


void gd_update_apply(float *y, float *u, float alpha, int n_samples, int d_out) {
	int N = (n_samples + 7) & (-1 ^ 7);
	__m256 yi, ui, alphas = _mm256_set1_ps(alpha);
	for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
		for (int i = 0; i < N; i += 16) {
			yi = _mm256_load_ps(y + kN + i);
			ui = _mm256_load_ps(u + kN + i);
			yi += ui;
			ui *= alphas;
			_mm256_store_ps(y + kN + i, yi);
			_mm256_store_ps(u + kN + i, ui);
		}
	}
}

#endif //GD_H
