#ifndef PRE_H
#define PRE_H

void pre_pair_sq_dist(float *p, float *dummy0, float *x, float *temp_n, int n_samples, int d_in) {
	float a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7,
			c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
			norma0, norma1, normb0, normb1, normb2, normb3, normb4, normb5, normb6, normb7;
	int N = (n_samples + 15) & (-1 ^ 15), D = (d_in + 15) & (-1 ^ 15), Bi, Bj;
	const int B0 = 128, B1 = 512;
	for (int i = 0, iD = 0; i < N; i += 8, iD += D * 8) {
		c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0.f;
		for (int k = 0; k < d_in; ++k) {
			a0 = x[iD + k];
			a1 = x[iD + D + k];
			a2 = x[iD + D * 2 + k];
			a3 = x[iD + D * 3 + k];
			a4 = x[iD + D * 4 + k];
			a5 = x[iD + D * 5 + k];
			a6 = x[iD + D * 6 + k];
			a7 = x[iD + D * 7 + k];
			c0 += a0 * a0;
			c1 += a1 * a1;
			c2 += a2 * a2;
			c3 += a3 * a3;
			c4 += a4 * a4;
			c5 += a5 * a5;
			c6 += a6 * a6;
			c7 += a7 * a7;
		}
		temp_n[i] = c0;
		temp_n[i + 1] = c1;
		temp_n[i + 2] = c2;
		temp_n[i + 3] = c3;
		temp_n[i + 4] = c4;
		temp_n[i + 5] = c5;
		temp_n[i + 6] = c6;
		temp_n[i + 7] = c7;
	}
	for (int J = 0; J < N; J += B1) {
		Bj = std::min(n_samples, J + B1);
		for (int I = J; I < N; I += B0) {
			Bi = std::min(n_samples, I + B0);
			for (int j = J, jD = J * D; j < Bj; j += 8, jD += D * 8) {
				normb0 = temp_n[j];
				normb1 = temp_n[j + 1];
				normb2 = temp_n[j + 2];
				normb3 = temp_n[j + 3];
				normb4 = temp_n[j + 4];
				normb5 = temp_n[j + 5];
				normb6 = temp_n[j + 6];
				normb7 = temp_n[j + 7];
				for (int i = std::max(j, I), iD = i * D, iN = i * N;
				     i < Bi; i += 2, iD += D * 2, iN += N * 2) {
					c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = c10 = c11 = c12 = c13 = c14 = c15 = 0.f;
					for (int k = 0; k < d_in; ++k) {
						a0 = x[iD + k];
						a1 = x[iD + D + k];
						b0 = x[jD + k];
						b1 = x[jD + D + k];
						b2 = x[jD + D * 2 + k];
						b3 = x[jD + D * 3 + k];
						b4 = x[jD + D * 4 + k];
						b5 = x[jD + D * 5 + k];
						b6 = x[jD + D * 6 + k];
						b7 = x[jD + D * 7 + k];
						c0 += a0 * b0;
						c1 += a0 * b1;
						c2 += a0 * b2;
						c3 += a0 * b3;
						c4 += a0 * b4;
						c5 += a0 * b5;
						c6 += a0 * b6;
						c7 += a0 * b7;
						c8 += a1 * b0;
						c9 += a1 * b1;
						c10 += a1 * b2;
						c11 += a1 * b3;
						c12 += a1 * b4;
						c13 += a1 * b5;
						c14 += a1 * b6;
						c15 += a1 * b7;
					}
					norma0 = temp_n[i];
					norma1 = temp_n[i + 1];
					p[iN + j] = norma0 - 2.f * c0 + normb0;
					p[iN + j + 1] = norma0 - 2.f * c1 + normb1;
					p[iN + j + 2] = norma0 - 2.f * c2 + normb2;
					p[iN + j + 3] = norma0 - 2.f * c3 + normb3;
					p[iN + j + 4] = norma0 - 2.f * c4 + normb4;
					p[iN + j + 5] = norma0 - 2.f * c5 + normb5;
					p[iN + j + 6] = norma0 - 2.f * c6 + normb6;
					p[iN + j + 7] = norma0 - 2.f * c7 + normb7;
					p[iN + N + j] = norma1 - 2.f * c8 + normb0;
					p[iN + N + j + 1] = norma1 - 2.f * c9 + normb1;
					p[iN + N + j + 2] = norma1 - 2.f * c10 + normb2;
					p[iN + N + j + 3] = norma1 - 2.f * c11 + normb3;
					p[iN + N + j + 4] = norma1 - 2.f * c12 + normb4;
					p[iN + N + j + 5] = norma1 - 2.f * c13 + normb5;
					p[iN + N + j + 6] = norma1 - 2.f * c14 + normb6;
					p[iN + N + j + 7] = norma1 - 2.f * c15 + normb7;
				}
			}
		}
	}
	for (int i = N - 16, iN = i * N; i < n_samples; ++i, iN += N) {
		for (int j = n_samples; j < N; ++j) {
			p[iN + j] = 0;
		}
	}
	memset(p + n_samples * N, 0, N * N - n_samples * N);
}

void pre_unfold_low_tri(float *p, int n_samples) {
	int N = (n_samples + 15) & (-1 ^ 15), Bi, Bj;
	for (int I = 0; I < N; I += 16) {
		Bi = I + 16;
		for (int J = 0; J <= I; J += 16) {
			for (int i = I, iN = I * N; i < Bi; ++i, iN += N) {
				Bj = std::min(J + 16, i);
				for (int j = J, jN = J * N; j < Bj; ++j, jN += N) {
					p[jN + i] = p[iN + j];
				}
			}
		}
	}
}

// Approximation of the exponential function by linear regression (polynomial of order three)
inline float exp_app(float x) {
	// -1 <= x <= 0
	float y;
	y = (.10250045262707179f * x) + .4624692123106021f;
	y = (y * x) + .9920487460431511f;
	y = (y * x) + .9996136409397813f;
	return y;
}

int pre_perplex_bi_search(float *p, float *dummy0, float perplexity, float epsilon, float *temp_3n, int n_samples) {
	float dist, dist_max, e, el, er, sum_e, nsum_ed, ss;
	int N = (n_samples + 15) & (-1 ^ 15), mode, iter, count = 0;
	float beta, beta_l, beta_r, beta_last = 0.f, h,
			h_tar = logf(perplexity), h_tar_u = h_tar + epsilon, h_tar_l = h_tar - epsilon,
			*e_m = temp_3n, *e_l = e_m + N, *e_r = e_l + N;
	// beta := -.5f / (sigma * sigma)
	bool ub_l, ub_r;
	for (int i = 0, iN = 0; i < n_samples; ++i, iN += N) {
		dist_max = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; ++j) {
			dist_max = std::max(p[iN + j], dist_max);
		}
		beta = -1.f / dist_max;
		mode = 0;
		ub_l = ub_r = true;
		for (iter = 0; beta != beta_last;) {
			++iter;
			beta_last = beta;
			sum_e = nsum_ed = 0.f;
			switch (mode) {
				case 0:
					for (int j = 0; j < n_samples; ++j) {
						dist = p[iN + j];
						e = dist ? exp_app(dist * beta) : 0.f; // assume data points are unique
						e_m[j] = e;
						sum_e += e;
						nsum_ed -= e * dist;
					}
					break;
				case 1:
					for (int j = 0; j < n_samples; ++j) {
						e = e_l[j];
						e *= e;
						e_m[j] = e;
						sum_e += e;
						nsum_ed -= e * p[iN + j];
					}
					break;
				case 2:
					for (int j = 0; j < n_samples; ++j) {
						e = sqrtf(e_r[j]);
						e_m[j] = e;
						sum_e += e;
						nsum_ed -= e * p[iN + j];
					}
					break;
				default:
					for (int j = 0; j < n_samples; ++j) {
						el = e_l[j];
						er = e_r[j];
						e = sqrtf(el * er);
						e_m[j] = e;
						sum_e += e;
						nsum_ed -= e * p[iN + j];
					}
			}
			h = nsum_ed * (beta / sum_e) + logf(sum_e);
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
		ss = 1.f / sum_e;
		for (int j = 0; j < n_samples; ++j) {
			e = e_m[j];
			p[iN + j] = e * ss;
		}
		count += iter;
	}
	return count;
}

void pre_sym_aff(float *p, int n_samples) {
	float k = .5f / (float) n_samples;
	int N = (n_samples + 15) & (-1 ^ 15), Bi, Bj;
	const int stair = 4; // default 4, change to 16 when use gd of avx512
	for (int I = 0; I < N; I += 16) {
		Bi = I + 16;
		for (int J = 0; J <= I; J += 16) {
			for (int i = I, iN = I * N; i < Bi; ++i, iN += N) {
				Bj = std::min(J + 16, i);
				for (int j = J, jN = J * N; j < Bj; ++j, jN += N) {
					p[iN + j] = (p[iN + j] + p[jN + i]) * k;
				}
			}
		}
	}
	for (int I = 0; I < N; I += stair) {
		Bi = I + stair;
		for (int i = I, iN = I * N; i < Bi; ++i, iN += N) {
			for (int j = I, jN = I * N; j < i; ++j, jN += N) {
				p[jN + i] = p[iN + j];
			}
		}
	}
}

#endif //PRE_H
