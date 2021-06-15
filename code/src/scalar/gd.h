#ifndef GD_H
#define GD_H

float gd_pair_aff(float *t, float *y, int n_samples, int d_out) {
	float y_i0, y_i1, y_i2, y_i3, y_j0, y_j1, y_j2, y_j3,
			diff_i0j0, diff_i0j1, diff_i0j2, diff_i0j3, diff_i1j0, diff_i1j1, diff_i1j2, diff_i1j3,
			diff_i2j0, diff_i2j1, diff_i2j2, diff_i2j3, diff_i3j0, diff_i3j1, diff_i3j2, diff_i3j3,
			t_i0j0, t_i0j1, t_i0j2, t_i0j3, t_i1j0, t_i1j1, t_i1j2, t_i1j3,
			t_i2j0, t_i2j1, t_i2j2, t_i2j3, t_i3j0, t_i3j1, t_i3j2, t_i3j3,
			t_sum = 0.f, c, csub = 0.f;
	int N = (n_samples + 15) & (-1 ^ 15), Nb = (n_samples + 3) & (-1 ^ 3), Bi, Bj;
	const int B = 16;
	for (int I = 0; I < N; I += B) {
		Bi = std::min(n_samples, I + B);
		for (int J = 0; J <= I; J += B) {
			for (int i = I, iN = i * N; i < Bi; i += 4, iN += N * 4) {
				Bj = std::min(i + 1, J + B);
				for (int j = J; j < Bj; j += 4) {
					t_i0j0 = t_i0j1 = t_i0j2 = t_i0j3 = t_i1j0 = t_i1j1 = t_i1j2 = t_i1j3 =
					t_i2j0 = t_i2j1 = t_i2j2 = t_i2j3 = t_i3j0 = t_i3j1 = t_i3j2 = t_i3j3 = 1.f;
					for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
						y_i0 = y[kN + i];
						y_i1 = y[kN + i + 1];
						y_i2 = y[kN + i + 2];
						y_i3 = y[kN + i + 3];
						y_j0 = y[kN + j];
						y_j1 = y[kN + j + 1];
						y_j2 = y[kN + j + 2];
						y_j3 = y[kN + j + 3];
						diff_i0j0 = y_i0 - y_j0;
						diff_i0j1 = y_i0 - y_j1;
						diff_i0j2 = y_i0 - y_j2;
						diff_i0j3 = y_i0 - y_j3;
						diff_i1j0 = y_i1 - y_j0;
						diff_i1j1 = y_i1 - y_j1;
						diff_i1j2 = y_i1 - y_j2;
						diff_i1j3 = y_i1 - y_j3;
						diff_i2j0 = y_i2 - y_j0;
						diff_i2j1 = y_i2 - y_j1;
						diff_i2j2 = y_i2 - y_j2;
						diff_i2j3 = y_i2 - y_j3;
						diff_i3j0 = y_i3 - y_j0;
						diff_i3j1 = y_i3 - y_j1;
						diff_i3j2 = y_i3 - y_j2;
						diff_i3j3 = y_i3 - y_j3;
						t_i0j0 += diff_i0j0 * diff_i0j0;
						t_i0j1 += diff_i0j1 * diff_i0j1;
						t_i0j2 += diff_i0j2 * diff_i0j2;
						t_i0j3 += diff_i0j3 * diff_i0j3;
						t_i1j0 += diff_i1j0 * diff_i1j0;
						t_i1j1 += diff_i1j1 * diff_i1j1;
						t_i1j2 += diff_i1j2 * diff_i1j2;
						t_i1j3 += diff_i1j3 * diff_i1j3;
						t_i2j0 += diff_i2j0 * diff_i2j0;
						t_i2j1 += diff_i2j1 * diff_i2j1;
						t_i2j2 += diff_i2j2 * diff_i2j2;
						t_i2j3 += diff_i2j3 * diff_i2j3;
						t_i3j0 += diff_i3j0 * diff_i3j0;
						t_i3j1 += diff_i3j1 * diff_i3j1;
						t_i3j2 += diff_i3j2 * diff_i3j2;
						t_i3j3 += diff_i3j3 * diff_i3j3;
					}
					t_i0j0 = 1.f / t_i0j0;
					t_i0j1 = 1.f / t_i0j1;
					t_i0j2 = 1.f / t_i0j2;
					t_i0j3 = 1.f / t_i0j3;
					t_i1j0 = 1.f / t_i1j0;
					t_i1j1 = 1.f / t_i1j1;
					t_i1j2 = 1.f / t_i1j2;
					t_i1j3 = 1.f / t_i1j3;
					t_i2j0 = 1.f / t_i2j0;
					t_i2j1 = 1.f / t_i2j1;
					t_i2j2 = 1.f / t_i2j2;
					t_i2j3 = 1.f / t_i2j3;
					t_i3j0 = 1.f / t_i3j0;
					t_i3j1 = 1.f / t_i3j1;
					t_i3j2 = 1.f / t_i3j2;
					t_i3j3 = 1.f / t_i3j3;
					t[iN + j] = t_i0j0;
					t[iN + j + 1] = t_i0j1;
					t[iN + j + 2] = t_i0j2;
					t[iN + j + 3] = t_i0j3;
					t[iN + N + j] = t_i1j0;
					t[iN + N + j + 1] = t_i1j1;
					t[iN + N + j + 2] = t_i1j2;
					t[iN + N + j + 3] = t_i1j3;
					t[iN + N * 2 + j] = t_i2j0;
					t[iN + N * 2 + j + 1] = t_i2j1;
					t[iN + N * 2 + j + 2] = t_i2j2;
					t[iN + N * 2 + j + 3] = t_i2j3;
					t[iN + N * 3 + j] = t_i3j0;
					t[iN + N * 3 + j + 1] = t_i3j1;
					t[iN + N * 3 + j + 2] = t_i3j2;
					t[iN + N * 3 + j + 3] = t_i3j3;
					c = (((t_i0j0 + t_i0j1) + (t_i0j2 + t_i0j3)) + ((t_i1j0 + t_i1j1) + (t_i1j2 + t_i1j3))) +
					    (((t_i2j0 + t_i2j1) + (t_i2j2 + t_i2j3)) + ((t_i3j0 + t_i3j1) + (t_i3j2 + t_i3j3)));
					if (i != j) {
						t_sum += c * 2.f;
					} else {
						t_sum += c;
					}
				}
			}
		}
	}
	for (int i = Nb - 4, iN = i * N; i < n_samples; ++i, iN += N) {
		for (int j = n_samples; j < N; ++j) {
			t[iN + j] = 0;
		}
	}
	for (int i = n_samples, iN = n_samples * N; i < Nb; ++i, iN += N) {
		for (int j = 0; j < Nb; ++j) {
			csub += t[i * N + j];
			t[i * N + j] = 0;
		}
	}
	t_sum -= 2.f * csub - (float) ((Nb - n_samples) * (Nb - n_samples) - n_samples);
	return 1.f / t_sum;
}

void gd_update_calc(float *u, float *dummy0, float *y, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	float y_i0, y_i1, y_i2, y_i3, y_j0, y_j1, y_j2, y_j3,
			c_i0j0, c_i0j1, c_i0j2, c_i0j3, c_i1j0, c_i1j1, c_i1j2, c_i1j3,
			c_i2j0, c_i2j1, c_i2j2, c_i2j3, c_i3j0, c_i3j1, c_i3j2, c_i3j3,
			g_i0j0, g_i0j1, g_i0j2, g_i0j3, g_i1j0, g_i1j1, g_i1j2, g_i1j3,
			g_i2j0, g_i2j1, g_i2j2, g_i2j3, g_i3j0, g_i3j1, g_i3j2, g_i3j3,
			p_i0j0, p_i0j1, p_i0j2, p_i0j3, p_i1j0, p_i1j1, p_i1j2, p_i1j3,
			p_i2j0, p_i2j1, p_i2j2, p_i2j3, p_i3j0, p_i3j1, p_i3j2, p_i3j3,
			t_i0j0, t_i0j1, t_i0j2, t_i0j3, t_i1j0, t_i1j1, t_i1j2, t_i1j3,
			t_i2j0, t_i2j1, t_i2j2, t_i2j3, t_i3j0, t_i3j1, t_i3j2, t_i3j3;
	int N = (n_samples + 15) & (-1 ^ 15), Bi, Bj;
	const int B = 16;
	for (int I = 0; I < N; I += B) {
		Bi = std::min(n_samples, I + B);
		for (int J = 0; J <= I; J += B) {
			for (int i = I, iN = i * N; i < Bi; i += 4, iN += N * 4) {
				Bj = std::min(i + 1, J + B);
				for (int j = J; j < Bj; j += 4) {
					t_i0j0 = t[iN + j];
					t_i0j1 = t[iN + j + 1];
					t_i0j2 = t[iN + j + 2];
					t_i0j3 = t[iN + j + 3];
					t_i1j0 = t[iN + N + j];
					t_i1j1 = t[iN + N + j + 1];
					t_i1j2 = t[iN + N + j + 2];
					t_i1j3 = t[iN + N + j + 3];
					t_i2j0 = t[iN + N * 2 + j];
					t_i2j1 = t[iN + N * 2 + j + 1];
					t_i2j2 = t[iN + N * 2 + j + 2];
					t_i2j3 = t[iN + N * 2 + j + 3];
					t_i3j0 = t[iN + N * 3 + j];
					t_i3j1 = t[iN + N * 3 + j + 1];
					t_i3j2 = t[iN + N * 3 + j + 2];
					t_i3j3 = t[iN + N * 3 + j + 3];
					p_i0j0 = p[iN + j];
					p_i0j1 = p[iN + j + 1];
					p_i0j2 = p[iN + j + 2];
					p_i0j3 = p[iN + j + 3];
					p_i1j0 = p[iN + N + j];
					p_i1j1 = p[iN + N + j + 1];
					p_i1j2 = p[iN + N + j + 2];
					p_i1j3 = p[iN + N + j + 3];
					p_i2j0 = p[iN + N * 2 + j];
					p_i2j1 = p[iN + N * 2 + j + 1];
					p_i2j2 = p[iN + N * 2 + j + 2];
					p_i2j3 = p[iN + N * 2 + j + 3];
					p_i3j0 = p[iN + N * 3 + j];
					p_i3j1 = p[iN + N * 3 + j + 1];
					p_i3j2 = p[iN + N * 3 + j + 2];
					p_i3j3 = p[iN + N * 3 + j + 3];
					c_i0j0 = (p_i0j0 - t_i0j0 * t_sum_inv) * t_i0j0;
					c_i0j1 = (p_i0j1 - t_i0j1 * t_sum_inv) * t_i0j1;
					c_i0j2 = (p_i0j2 - t_i0j2 * t_sum_inv) * t_i0j2;
					c_i0j3 = (p_i0j3 - t_i0j3 * t_sum_inv) * t_i0j3;
					c_i1j0 = (p_i1j0 - t_i1j0 * t_sum_inv) * t_i1j0;
					c_i1j1 = (p_i1j1 - t_i1j1 * t_sum_inv) * t_i1j1;
					c_i1j2 = (p_i1j2 - t_i1j2 * t_sum_inv) * t_i1j2;
					c_i1j3 = (p_i1j3 - t_i1j3 * t_sum_inv) * t_i1j3;
					c_i2j0 = (p_i2j0 - t_i2j0 * t_sum_inv) * t_i2j0;
					c_i2j1 = (p_i2j1 - t_i2j1 * t_sum_inv) * t_i2j1;
					c_i2j2 = (p_i2j2 - t_i2j2 * t_sum_inv) * t_i2j2;
					c_i2j3 = (p_i2j3 - t_i2j3 * t_sum_inv) * t_i2j3;
					c_i3j0 = (p_i3j0 - t_i3j0 * t_sum_inv) * t_i3j0;
					c_i3j1 = (p_i3j1 - t_i3j1 * t_sum_inv) * t_i3j1;
					c_i3j2 = (p_i3j2 - t_i3j2 * t_sum_inv) * t_i3j2;
					c_i3j3 = (p_i3j3 - t_i3j3 * t_sum_inv) * t_i3j3;
					for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
						y_i0 = y[kN + i];
						y_i1 = y[kN + i + 1];
						y_i2 = y[kN + i + 2];
						y_i3 = y[kN + i + 3];
						y_j0 = y[kN + j];
						y_j1 = y[kN + j + 1];
						y_j2 = y[kN + j + 2];
						y_j3 = y[kN + j + 3];
						g_i0j0 = (y_i0 - y_j0) * c_i0j0;
						g_i0j1 = (y_i0 - y_j1) * c_i0j1;
						g_i0j2 = (y_i0 - y_j2) * c_i0j2;
						g_i0j3 = (y_i0 - y_j3) * c_i0j3;
						g_i1j0 = (y_i1 - y_j0) * c_i1j0;
						g_i1j1 = (y_i1 - y_j1) * c_i1j1;
						g_i1j2 = (y_i1 - y_j2) * c_i1j2;
						g_i1j3 = (y_i1 - y_j3) * c_i1j3;
						g_i2j0 = (y_i2 - y_j0) * c_i2j0;
						g_i2j1 = (y_i2 - y_j1) * c_i2j1;
						g_i2j2 = (y_i2 - y_j2) * c_i2j2;
						g_i2j3 = (y_i2 - y_j3) * c_i2j3;
						g_i3j0 = (y_i3 - y_j0) * c_i3j0;
						g_i3j1 = (y_i3 - y_j1) * c_i3j1;
						g_i3j2 = (y_i3 - y_j2) * c_i3j2;
						g_i3j3 = (y_i3 - y_j3) * c_i3j3;
						u[kN + j] += g_i0j0 + g_i1j0 + g_i2j0 + g_i3j0;
						u[kN + j + 1] += g_i0j1 + g_i1j1 + g_i2j1 + g_i3j1;
						u[kN + j + 2] += g_i0j2 + g_i1j2 + g_i2j2 + g_i3j2;
						u[kN + j + 3] += g_i0j3 + g_i1j3 + g_i2j3 + g_i3j3;
						if (i != j) {
							u[kN + i] -= g_i0j0 + g_i0j1 + g_i0j2 + g_i0j3;
							u[kN + i + 1] -= g_i1j0 + g_i1j1 + g_i1j2 + g_i1j3;
							u[kN + i + 2] -= g_i2j0 + g_i2j1 + g_i2j2 + g_i2j3;
							u[kN + i + 3] -= g_i3j0 + g_i3j1 + g_i3j2 + g_i3j3;
						}
					}
				}
			}
		}
	}
}

void gd_update_apply(float *y, float *u, float *dummy0, float etax4, float alpha, int n_samples, int d_out) {
	int N = (n_samples + 15) & (-1 ^ 15);
	float u_ik;
	for (int k = 0, kN = 0; k < d_out; ++k, kN += N) {
		for (int i = 0; i < n_samples; ++i) {
			u_ik = u[kN + i];
			y[kN + i] += u_ik * etax4;
			u[kN + i] = u_ik * alpha;
		}
	}
}

#endif //GD_H
