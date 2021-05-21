#ifndef PRE_H
#define PRE_H

#include <cmath>
#include "block16.h"
#include "exp.h"

void pre_pair_sq_dist(float *p, float *x, int n_samples, int d_in) {
	// p: lower 16-blocked triangle
	// TODO: may try 8x2 or 4x4 blocking
	__m512 a, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, zerofs = _mm512_setzero_ps(),
			c, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
	int N = (n_samples + 15) & (-1 ^ 15), D = (d_in + 15) & (-1 ^ 15);
	for (int j = 0, jD = 0; j < N; j += 16, jD += D * 16) {
		for (int i = j, iD = jD, iN = j * N; i < n_samples; ++i, iD += D, iN += N) {
			c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = zerofs;
			for (int k = 0; k < D; k += 16) {
				a = _mm512_load_ps(x + iD + k);
				block16_load(x + jD + k, D, b0, b1, b2, b3, b4, b5, b6, b7,
				           b8, b9, b10, b11, b12, b13, b14, b15);
				b0 -= a;
				c0 = _mm512_fmadd_ps(b0, b0, c0);
				b1 -= a;
				c1 = _mm512_fmadd_ps(b1, b1, c1);
				b2 -= a;
				c2 = _mm512_fmadd_ps(b2, b2, c2);
				b3 -= a;
				c3 = _mm512_fmadd_ps(b3, b3, c3);
				b4 -= a;
				c4 = _mm512_fmadd_ps(b4, b4, c4);
				b5 -= a;
				c5 = _mm512_fmadd_ps(b5, b5, c5);
				b6 -= a;
				c6 = _mm512_fmadd_ps(b6, b6, c6);
				b7 -= a;
				c7 = _mm512_fmadd_ps(b7, b7, c7);
				b8 -= a;
				c8 = _mm512_fmadd_ps(b8, b8, c8);
				b9 -= a;
				c9 = _mm512_fmadd_ps(b9, b9, c9);
				b10 -= a;
				c10 = _mm512_fmadd_ps(b10, b10, c10);
				b11 -= a;
				c11 = _mm512_fmadd_ps(b11, b11, c11);
				b12 -= a;
				c12 = _mm512_fmadd_ps(b12, b12, c12);
				b13 -= a;
				c13 = _mm512_fmadd_ps(b13, b13, c13);
				b14 -= a;
				c14 = _mm512_fmadd_ps(b14, b14, c14);
				b15 -= a;
				c15 = _mm512_fmadd_ps(b15, b15, c15);
			}
			c = block16_row_sum(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15);
			_mm512_store_ps(p + iN + j, c);
		}
	}
	for (int i = N - 16, iN = i * N; i < n_samples; ++i, iN += N) {
		for (int j = n_samples; j < N; ++j) {
			p[iN + j] = 0;
		}
	}
	int limit = N * N;
	for (int offset = n_samples * N; offset < limit; offset += 16) {
		_mm512_store_ps(p + offset, zerofs);
	}
}

void pre_unfold_low_tri(float *p, int n_samples) {
	__m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 16, iN = i * N; i < N; i += 16, iN += N * 16) {
		for (int j = 0, jN = 0; j < i; j += 16, jN += N * 16) {
			block16_load(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
			           r8, r9, r10, r11, r12, r13, r14, r15);
			block16_transpose(r0, r1, r2, r3, r4, r5, r6, r7,
			                r8, r9, r10, r11, r12, r13, r14, r15);
			block16_store(p + jN + i, N, r0, r1, r2, r3, r4, r5, r6, r7,
			            r8, r9, r10, r11, r12, r13, r14, r15);
		}
	}
}

int pre_perplex_bi_search(float *p, float perplexity, float epsilon, int max_iter, float *temp_3n, int n_samples) {
	__m512 d, d_max, e, el, er, betas, sum_e, nsum_edb, ss, zerofs = _mm512_setzero_ps();
	__mmask16 mask;
	int N = (n_samples + 15) & (-1 ^ 15), mode, iter, count = 0;
	float beta, beta_l, beta_r, beta_last = 0.f, h, s,
			h_tar = logf(perplexity), h_tar_u = h_tar + epsilon, h_tar_l = h_tar - epsilon,
			*e_m = temp_3n, *e_l = e_m + N, *e_r = e_l + N;
	// beta = -.5f / (sigma * sigma)
	bool ub_l, ub_r;
	for (int i = 0, iN = 0; i < n_samples; ++i, iN += N) {
		d_max = zerofs;
		for (int j = 0; j < N; j += 16) { // TODO: loop unrolling
			d = _mm512_load_ps(p + iN + j);
			d_max = _mm512_max_ps(d, d_max);
		}
		beta = -1.f / _mm512_reduce_max_ps(d_max);
		mode = 0;
		ub_l = ub_r = true;
		for (iter = 1; beta != beta_last && iter <= max_iter; ++iter) {
			beta_last = beta;
			sum_e = nsum_edb = zerofs;
			betas = _mm512_set1_ps(beta);
			for (int j = 0; j < N; j += 16) { // TODO: loop unrolling
				switch (mode) {
					case 0:
						d = _mm512_load_ps(p + iN + j);
						mask = _mm512_test_epi32_mask((__m512i) d, (__m512i) d);
						e = exp_app_ps(d * betas);
						e = _mm512_maskz_mov_ps(mask, e); // assume data points are unique
						break;
					case 1:
						e = _mm512_load_ps(e_l + j);
						e *= e;
						break;
					case 2:
						e = _mm512_load_ps(e_r + j);
						e = _mm512_sqrt_ps(e);
						break;
					default:
						el = _mm512_load_ps(e_l + j);
						er = _mm512_load_ps(e_r + j);
						e = _mm512_sqrt_ps(el * er);
				}
				_mm512_store_ps(e_m + j, e);
				sum_e += e;
				d = _mm512_load_ps(p + iN + j);
				nsum_edb = _mm512_fnmadd_ps(e, d * betas, nsum_edb);
			}
			s = _mm512_reduce_add_ps(sum_e);
			h = _mm512_reduce_add_ps(nsum_edb) / s + logf(s);
			if (h > h_tar_u) {
				beta_l = beta;
				ub_l = false;
				std::swap(e_m, e_l);
				if (ub_r) {
					beta *= 2.f;
					mode = 1;
				} else {
					beta = (beta_l + beta_r) * .5f;
					mode = 3;
				}
			} else if (h < h_tar_l) {
				beta_r = beta;
				ub_r = false;
				std::swap(e_m, e_r);
				if (ub_l) {
					beta *= .5f;
					mode = 2;
				} else {
					beta = (beta_l + beta_r) * .5f;
					mode = 3;
				}
			} else {
				break;
			}
		}
		ss = _mm512_set1_ps(1.f / s);
		for (int j = 0; j < N; j += 16) {
			e = _mm512_load_ps(e_m + j);
			_mm512_store_ps(p + iN + j, e * ss);
		}
		count += iter;
	}
	return count;
}

void pre_sym_aff_ex(float *p, float *p_ex, float ex_rate, int n_samples) {
	__m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
			c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
			k = _mm512_set1_ps(.5f / (float) n_samples), ex_rates = _mm512_set1_ps(ex_rate);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0, iN = i * N; i < N; i += 16, iN += N * 16) {
		for (int j = 0, jN = 0; j <= i; j += 16, jN += N * 16) {
			block16_load(p + jN + i, N, c0, c1, c2, c3, c4, c5, c6, c7,
			           c8, c9, c10, c11, c12, c13, c14, c15);
			block16_transpose(c0, c1, c2, c3, c4, c5, c6, c7,
			                c8, c9, c10, c11, c12, c13, c14, c15);
			block16_load(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
			           r8, r9, r10, r11, r12, r13, r14, r15);
			r0 = (r0 + c0) * k;
			r1 = (r1 + c1) * k;
			r2 = (r2 + c2) * k;
			r3 = (r3 + c3) * k;
			r4 = (r4 + c4) * k;
			r5 = (r5 + c5) * k;
			r6 = (r6 + c6) * k;
			r7 = (r7 + c7) * k;
			r8 = (r8 + c8) * k;
			r9 = (r9 + c9) * k;
			r10 = (r10 + c10) * k;
			r11 = (r11 + c11) * k;
			r12 = (r12 + c12) * k;
			r13 = (r13 + c13) * k;
			r14 = (r14 + c14) * k;
			r15 = (r15 + c15) * k;
			block16_store(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
			            r8, r9, r10, r11, r12, r13, r14, r15);
			if (p_ex) {
				r0 *= ex_rates;
				r1 *= ex_rates;
				r2 *= ex_rates;
				r3 *= ex_rates;
				r4 *= ex_rates;
				r5 *= ex_rates;
				r6 *= ex_rates;
				r7 *= ex_rates;
				r8 *= ex_rates;
				r9 *= ex_rates;
				r10 *= ex_rates;
				r11 *= ex_rates;
				r12 *= ex_rates;
				r13 *= ex_rates;
				r14 *= ex_rates;
				r15 *= ex_rates;
				block16_store(p_ex + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
				            r8, r9, r10, r11, r12, r13, r14, r15);
			}
		}
	}
}

#endif //PRE_H
