#ifndef PRE_H
#define PRE_H

#include <cmath>
#include "block.h"
#include "exp.h"

void pre_pair_sq_dist(float *p, float *x, float *temp_n, int n_samples, int d_in) {
	__m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
			b0, b1, b2, b3, b4, b5, b6, b7, zerofs = _mm512_setzero_ps(),
			c, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15;
	__m256 cl, cu, norma0, norma1, normb, twos = _mm256_set1_ps(2.f);
	int N = (n_samples + 15) & (-1 ^ 15), D = (d_in + 15) & (-1 ^ 15), Bi, Bj;
	const int B0 = 128, B1 = 512;
	for (int i = 0, iD = 0; i < N; i += 16, iD += D * 16) {
		c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = zerofs;
		for (int k = 0; k < D; k += 16) {
			block_load(x + iD + k, D, a0, a1, a2, a3, a4, a5, a6, a7,
			           a8, a9, a10, a11, a12, a13, a14, a15);
			c0 = _mm512_fmadd_ps(a0, a0, c0);
			c1 = _mm512_fmadd_ps(a1, a1, c1);
			c2 = _mm512_fmadd_ps(a2, a2, c2);
			c3 = _mm512_fmadd_ps(a3, a3, c3);
			c4 = _mm512_fmadd_ps(a4, a4, c4);
			c5 = _mm512_fmadd_ps(a5, a5, c5);
			c6 = _mm512_fmadd_ps(a6, a6, c6);
			c7 = _mm512_fmadd_ps(a7, a7, c7);
			c8 = _mm512_fmadd_ps(a8, a8, c8);
			c9 = _mm512_fmadd_ps(a9, a9, c9);
			c10 = _mm512_fmadd_ps(a10, a10, c10);
			c11 = _mm512_fmadd_ps(a11, a11, c11);
			c12 = _mm512_fmadd_ps(a12, a12, c12);
			c13 = _mm512_fmadd_ps(a13, a13, c13);
			c14 = _mm512_fmadd_ps(a14, a14, c14);
			c15 = _mm512_fmadd_ps(a15, a15, c15);
		}
		c = block_row_sum(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15);
		_mm512_store_ps(temp_n + i, c);
	}
	for (int J = 0; J < N; J += B1) {
		Bj = std::min(n_samples, J + B1);
		for (int I = J; I < N; I += B0) {
			Bi = std::min(n_samples, I + B0);
			for (int j = J, jD = J * D; j < Bj; j += 8, jD += D * 8) {
				normb = _mm256_load_ps(temp_n + j);
				for (int i = std::max(j, I), iD = i * D, iN = i * N;
				     i < Bi; i += 2, iD += D * 2, iN += N * 2) {
					c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = zerofs;
					for (int k = 0; k < D; k += 16) {
						a0 = _mm512_load_ps(x + iD + k);
						a1 = _mm512_load_ps(x + iD + D + k);
						block_load(x + jD + k, D, b0, b1, b2, b3, b4, b5, b6, b7);
						c0 = _mm512_fmadd_ps(a0, b0, c0);
						c1 = _mm512_fmadd_ps(a0, b1, c1);
						c2 = _mm512_fmadd_ps(a0, b2, c2);
						c3 = _mm512_fmadd_ps(a0, b3, c3);
						c4 = _mm512_fmadd_ps(a0, b4, c4);
						c5 = _mm512_fmadd_ps(a0, b5, c5);
						c6 = _mm512_fmadd_ps(a0, b6, c6);
						c7 = _mm512_fmadd_ps(a0, b7, c7);
						c8 = _mm512_fmadd_ps(a1, b0, c8);
						c9 = _mm512_fmadd_ps(a1, b1, c9);
						c10 = _mm512_fmadd_ps(a1, b2, c10);
						c11 = _mm512_fmadd_ps(a1, b3, c11);
						c12 = _mm512_fmadd_ps(a1, b4, c12);
						c13 = _mm512_fmadd_ps(a1, b5, c13);
						c14 = _mm512_fmadd_ps(a1, b6, c14);
						c15 = _mm512_fmadd_ps(a1, b7, c15);
					}
					c = block_row_sum(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15);
					cl = _mm512_castps512_ps256(c);
					cu = _mm512_extractf32x8_ps(c, 1);
					norma0 = _mm256_broadcast_ss(temp_n + i);
					norma1 = _mm256_broadcast_ss(temp_n + i + 1);
					cl = _mm256_fnmadd_ps(cl, twos, norma0) + normb;
					cu = _mm256_fnmadd_ps(cu, twos, norma1) + normb;
					_mm256_store_ps(p + iN + j, cl);
					_mm256_store_ps(p + iN + N + j, cu);
				}
			}
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
	__m256 rr0, rr1, rr2, rr3, rr4, rr5, rr6, rr7;
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 16, iN = i * N; i < N; i += 16, iN += N * 16) {
		for (int j = 0, jN = 0; j < i; j += 16, jN += N * 16) {
			block_load(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
			           r8, r9, r10, r11, r12, r13, r14, r15);
			block_transpose(r0, r1, r2, r3, r4, r5, r6, r7,
			                r8, r9, r10, r11, r12, r13, r14, r15);
			block_store(p + jN + i, N, r0, r1, r2, r3, r4, r5, r6, r7,
			            r8, r9, r10, r11, r12, r13, r14, r15);
		}
	}
	for (int i = 8, iN = i * N; i < N - 8; i += 16, iN += N * 16) {
		block_load(p + iN + i - 8, N, rr0, rr1, rr2, rr3, rr4, rr5, rr6, rr7);
		block_transpose(rr0, rr1, rr2, rr3, rr4, rr5, rr6, rr7);
		block_store(p + iN - N * 8 + i, N, rr0, rr1, rr2, rr3, rr4, rr5, rr6, rr7);
	}
}

int pre_perplex_bi_search(float *p, float perplexity, float epsilon, float *temp_3n, int n_samples) {
	__m512 dist, dist_max, e, el, er, betas, sum_e, nsum_ed, ss, zerofs = _mm512_setzero_ps();
	__mmask16 mask;
	int N = (n_samples + 15) & (-1 ^ 15), mode, iter, count = 0;
	float beta, beta_l, beta_r, beta_last = 0.f, h, s,
			h_tar = logf(perplexity), h_tar_u = h_tar + epsilon, h_tar_l = h_tar - epsilon,
			*e_m = temp_3n, *e_l = e_m + N, *e_r = e_l + N;
	// beta := -.5f / (sigma * sigma)
	bool ub_l, ub_r;
	for (int i = 0, iN = 0; i < n_samples; ++i, iN += N) {
		dist_max = zerofs;
		for (int j = 0; j < N; j += 16) {
			dist = _mm512_load_ps(p + iN + j);
			dist_max = _mm512_max_ps(dist, dist_max);
		}
		beta = -1.f / _mm512_reduce_max_ps(dist_max);
		mode = 0;
		ub_l = ub_r = true;
		for (iter = 1; beta != beta_last; ++iter) {
			beta_last = beta;
			sum_e = nsum_ed = zerofs;
			for (int j = 0; j < N; j += 16) {
				switch (mode) {
					case 0:
						dist = _mm512_load_ps(p + iN + j);
						betas = _mm512_set1_ps(beta);
						e = exp_app_ps(dist * betas);
						mask = _mm512_test_epi32_mask((__m512i) dist, (__m512i) dist);
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
				dist = _mm512_load_ps(p + iN + j);
				nsum_ed = _mm512_fnmadd_ps(e, dist, nsum_ed);
			}
			s = _mm512_reduce_add_ps(sum_e);
			h = _mm512_reduce_add_ps(nsum_ed) * (beta / s) + logf(s);
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

void pre_sym_aff(float *p, int n_samples) {
	__m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
			c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
			k = _mm512_set1_ps(.5f / (float) n_samples);
	int N = (n_samples + 15) & (-1 ^ 15);
	for (int i = 0, iN = i * N; i < N; i += 16, iN += N * 16) {
		for (int j = 0, jN = 0; j <= i; j += 16, jN += N * 16) {
			block_load(p + jN + i, N, c0, c1, c2, c3, c4, c5, c6, c7,
			           c8, c9, c10, c11, c12, c13, c14, c15);
			block_transpose(c0, c1, c2, c3, c4, c5, c6, c7,
			                c8, c9, c10, c11, c12, c13, c14, c15);
			block_load(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
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
			block_store(p + iN + j, N, r0, r1, r2, r3, r4, r5, r6, r7,
			            r8, r9, r10, r11, r12, r13, r14, r15);
		}
	}
}

#endif //PRE_H
