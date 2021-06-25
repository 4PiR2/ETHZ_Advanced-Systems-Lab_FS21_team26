#ifndef GD_H
#define GD_H

#include "block.h"

float compute_t(float *y, float *t, int n_samples, int d_out) {
	float sum_t = 0.f;
	for (int i = 0; i < n_samples; i++) {
		t[i * n_samples + i] = 0.f;
		for (int j = i + 1; j < n_samples; j++) {
			float dist = 1.f;
			for (int k = 0; k < d_out; k++) {
				float diff = y[i * d_out + k] - y[j * d_out + k];
				dist += diff * diff;
			}
			float aff = 1.f / dist;
			t[j * n_samples + i] = t[i * n_samples + j] = aff;
			sum_t += aff;
		}
	}
	return .5f / sum_t;
}

float compute_t_trans(float *y_trans, float *t, int n_samples, int d_out) {
	float sum_t = 0.f;
	for (int i = 0; i < n_samples; i++) {
		t[i * n_samples + i] = 0.f;
		for (int j = i + 1; j < n_samples; j++) {
			float dist = 1.f;
			for (int k = 0; k < d_out; k++) {
				float diff = y_trans[k * n_samples + i] - y_trans[k * n_samples + j];
				dist += diff * diff;
			}
			float aff = 1.f / dist;
			t[j * n_samples + i] = t[i * n_samples + j] = aff;
			sum_t += aff;
		}
	}
	return .5f / sum_t;
}

float compute_t_trans_block(float *y_trans, float *t, int n_samples, int d_out) {
	float sum_t = 0.f;

	for (int j = 0; j < n_samples; j+=4) {
		t[j * n_samples + j] = 0.f;

		float dist01 = 1.f, dist02 = 1.f, dist03 = 1.f, 
		      dist04 = 1.f, dist05 = 1.f, dist06 = 1.f;

		for (int k = 0; k < d_out; k++) {
			float diff01 = y_trans[k * n_samples + j] - y_trans[k * n_samples + j + 1];
			float diff02 = y_trans[k * n_samples + j] - y_trans[k * n_samples + j + 2];
			float diff03 = y_trans[k * n_samples + j] - y_trans[k * n_samples + j + 3];
			float diff04 = y_trans[k * n_samples + j + 1] - y_trans[k * n_samples + j + 2];
			float diff05 = y_trans[k * n_samples + j + 1] - y_trans[k * n_samples + j + 3];
			float diff06 = y_trans[k * n_samples + j + 2] - y_trans[k * n_samples + j + 3];

			dist01 += diff01 * diff01;
			dist02 += diff02 * diff02;
			dist03 += diff03 * diff03;
			dist04 += diff04 * diff04;
			dist05 += diff05 * diff05;
			dist06 += diff06 * diff06;
		}

		float aff01 = 1.f / dist01;
		float aff02 = 1.f / dist02;
		float aff03 = 1.f / dist03;
		float aff04 = 1.f / dist04;
		float aff05 = 1.f / dist05;
		float aff06 = 1.f / dist06;

		t[(j+1) * n_samples + j] = aff01;
		t[(j+2) * n_samples + j] = aff02;
		t[(j+3) * n_samples + j] = aff03;
		t[(j+2) * n_samples + j + 1] = aff04;
		t[(j+3) * n_samples + j + 1] = aff05;
		t[(j+3) * n_samples + j + 2] = aff06;

		sum_t += aff01;
		sum_t += aff02;
		sum_t += aff03;
		sum_t += aff04;
		sum_t += aff05;
		sum_t += aff06;

		for (int i = j + 4; i < n_samples; i+=4) {
			float dist1 = 1.f, dist2 = 1.f, dist3 = 1.f, dist4 = 1.f, 
			      dist5 = 1.f, dist6 = 1.f, dist7 = 1.f, dist8 = 1.f,
				  dist9 = 1.f, dist10 = 1.f, dist11 = 1.f, dist12 = 1.f,
				  dist13 = 1.f, dist14 = 1.f, dist15 = 1.f, dist16 = 1.f;

			for (int k = 0; k < d_out; k++) {
				float diff1 = y_trans[k * n_samples + i] - y_trans[k * n_samples + j];
				float diff2 = y_trans[k * n_samples + i] - y_trans[k * n_samples + j + 1];
				float diff3 = y_trans[k * n_samples + i] - y_trans[k * n_samples + j + 2];
				float diff4 = y_trans[k * n_samples + i] - y_trans[k * n_samples + j + 3];

				float diff5 = y_trans[k * n_samples + i + 1] - y_trans[k * n_samples + j];
				float diff6 = y_trans[k * n_samples + i + 1] - y_trans[k * n_samples + j + 1];
				float diff7 = y_trans[k * n_samples + i + 1] - y_trans[k * n_samples + j + 2];
				float diff8 = y_trans[k * n_samples + i + 1] - y_trans[k * n_samples + j + 3];

				float diff9 = y_trans[k * n_samples + i + 2] - y_trans[k * n_samples + j];
				float diff10 = y_trans[k * n_samples + i + 2] - y_trans[k * n_samples + j + 1];
				float diff11 = y_trans[k * n_samples + i + 2] - y_trans[k * n_samples + j + 2];
				float diff12 = y_trans[k * n_samples + i + 2] - y_trans[k * n_samples + j + 3];

				float diff13 = y_trans[k * n_samples + i + 3] - y_trans[k * n_samples + j];
				float diff14 = y_trans[k * n_samples + i + 3] - y_trans[k * n_samples + j + 1];
				float diff15 = y_trans[k * n_samples + i + 3] - y_trans[k * n_samples + j + 2];
				float diff16 = y_trans[k * n_samples + i + 3] - y_trans[k * n_samples + j + 3];

				dist1 += diff1 * diff1;
				dist2 += diff2 * diff2;
				dist3 += diff3 * diff3;
				dist4 += diff4 * diff4;
				dist5 += diff5 * diff5;
				dist6 += diff6 * diff6;
				dist7 += diff7 * diff7;
				dist8 += diff8 * diff8;
				dist9 += diff9 * diff9;
				dist10 += diff10 * diff10;
				dist11 += diff11 * diff11;
				dist12 += diff12 * diff12;
				dist13 += diff13 * diff13;
				dist14 += diff14 * diff14;
				dist15 += diff15 * diff15;
				dist16 += diff16 * diff16;
			}
			float aff1 = 1.f / dist1;
			float aff2 = 1.f / dist2;
			float aff3 = 1.f / dist3;
			float aff4 = 1.f / dist4;
			float aff5 = 1.f / dist5;
			float aff6 = 1.f / dist6;
			float aff7 = 1.f / dist7;
			float aff8 = 1.f / dist8;
			float aff9 = 1.f / dist9;
			float aff10 = 1.f / dist10;
			float aff11 = 1.f / dist11;
			float aff12 = 1.f / dist12;
			float aff13 = 1.f / dist13;
			float aff14 = 1.f / dist14;
			float aff15 = 1.f / dist15;
			float aff16 = 1.f / dist16;

			t[i * n_samples + j] = aff1;
			t[i * n_samples + j + 1] = aff2;
			t[i * n_samples + j + 2] = aff3;
			t[i * n_samples + j + 3] = aff4;

			t[(i+1) * n_samples + j] = aff5;
			t[(i+1) * n_samples + j + 1] = aff6;
			t[(i+1) * n_samples + j + 2] = aff7;
			t[(i+1) * n_samples + j + 3] = aff8;

			t[(i+2) * n_samples + j] = aff9;
			t[(i+2) * n_samples + j + 1] = aff10;
			t[(i+2) * n_samples + j + 2] = aff11;
			t[(i+2) * n_samples + j + 3] = aff12;

			t[(i+3) * n_samples + j] = aff13;
			t[(i+3) * n_samples + j + 1] = aff14;
			t[(i+3) * n_samples + j + 2] = aff15;
			t[(i+3) * n_samples + j + 3] = aff16;

			sum_t += aff1;
			sum_t += aff2;
			sum_t += aff3;
			sum_t += aff4;

			sum_t += aff5;
			sum_t += aff6;
			sum_t += aff7;
			sum_t += aff8;

			sum_t += aff9;
			sum_t += aff10;
			sum_t += aff11;
			sum_t += aff12;

			sum_t += aff13;
			sum_t += aff14;
			sum_t += aff15;
			sum_t += aff16;

		}
	}

	return .5f / sum_t;
}

void gradientCompute(float *y, float *g, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float t_ij = t[i * n_samples + j], c_ij = (p[i * n_samples + j] - t_ij * t_sum_inv) * t_ij;
			for (int k = 0; k < d_out; k++) {
				float g_ijk = (y[i * d_out + k] - y[j * d_out + k]) * c_ij;
				g[i * d_out + k] += g_ijk;
				g[j * d_out + k] -= g_ijk;
			}
		}
	}
}

void gradientCompute_trans(float *y_trans, float *g, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float t_ij = t[i * n_samples + j], c_ij = (p[i * n_samples + j] - t_ij * t_sum_inv) * t_ij;
			for (int k = 0; k < d_out; k++) {
				float g_ijk = (y_trans[k * n_samples + i] - y_trans[k * n_samples + j]) * c_ij;
				g[k * n_samples + i] += g_ijk;
				g[k * n_samples + j] -= g_ijk;
			}
		}
	}
}

/*
void gradientCompute_trans(float *y_trans, float *g, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	for (int j = 0; j < n_samples; j++) {
		for (int i = j + 1; i < n_samples; i++) {
			float t_ij = t[i * n_samples + j], c_ij = (p[i * n_samples + j] - t_ij * t_sum_inv) * t_ij;
			for (int k = 0; k < d_out; k++) {
				float g_ijk = (y_trans[k * n_samples + i] - y_trans[k * n_samples + j]) * c_ij;
				g[k * n_samples + i] += g_ijk;
				g[k * n_samples + j] -= g_ijk;
			}
		}
	}
}
*/

void gradientCompute_trans_block(float *y_trans, float *g, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	for (int j = 0; j < n_samples; j+=4) {
		float t_ij01 = t[(j+1) * n_samples + j], t_ij02 = t[(j+2) * n_samples + j], t_ij03 = t[(j+2) * n_samples + j + 1], 
		      t_ij04 = t[(j+3) * n_samples + j], t_ij05 = t[(j+3) * n_samples + j + 1], t_ij06 = t[(j+3) * n_samples + j + 2];

		float p_ij01 = p[(j+1) * n_samples + j], p_ij02 = p[(j+2) * n_samples + j], p_ij03 = p[(j+2) * n_samples + j + 1], 
		      p_ij04 = p[(j+3) * n_samples + j], p_ij05 = p[(j+3) * n_samples + j + 1], p_ij06 = p[(j+3) * n_samples + j + 2];

		float c_ij01 = p_ij01, c_ij02 = p_ij02, c_ij03 = p_ij03, 
		      c_ij04 = p_ij04, c_ij05 = p_ij05, c_ij06 = p_ij06;

		c_ij01 -= t_ij01 * t_sum_inv;
		c_ij02 -= t_ij02 * t_sum_inv;
		c_ij03 -= t_ij03 * t_sum_inv;
		c_ij04 -= t_ij04 * t_sum_inv;
		c_ij05 -= t_ij05 * t_sum_inv;
		c_ij06 -= t_ij06 * t_sum_inv;

		c_ij01 *= t_ij01;
		c_ij02 *= t_ij02;
		c_ij03 *= t_ij03;
		c_ij04 *= t_ij04;
		c_ij05 *= t_ij05;
		c_ij06 *= t_ij06;

		for (int k = 0; k < d_out; k++) {
			float yjk01 = y_trans[k * n_samples + j], yjk02 = y_trans[k * n_samples + j + 1], yjk03 = y_trans[k * n_samples + j + 2], yjk04 = y_trans[k * n_samples + j + 3];

			float g_ijk01 = yjk02 - yjk01;
			float g_ijk02 = yjk03 - yjk01;
			float g_ijk03 = yjk03 - yjk02;
			float g_ijk04 = yjk04 - yjk01;
			float g_ijk05 = yjk04 - yjk02;
			float g_ijk06 = yjk04 - yjk03;

			g_ijk01 *= c_ij01;
			g_ijk02 *= c_ij02;
			g_ijk03 *= c_ij03;
			g_ijk04 *= c_ij04;
			g_ijk05 *= c_ij05;
			g_ijk06 *= c_ij06;

			g[k * n_samples + j + 1] += g_ijk01;
			g[k * n_samples + j + 2] += g_ijk02 + g_ijk03;
			g[k * n_samples + j + 3] += g_ijk04 + g_ijk05 + g_ijk06;

			g[k * n_samples + j] -= g_ijk01 + g_ijk02 + g_ijk04;
			g[k * n_samples + j + 1] -= g_ijk03 + g_ijk05;
			g[k * n_samples + j + 2] -= g_ijk06;
		}

		for (int i = j + 4; i < n_samples; i+=4) {
			float t_ij1 = t[i * n_samples + j], t_ij2 = t[i * n_samples + j + 1], t_ij3 = t[i * n_samples + j + 2], t_ij4 = t[i * n_samples + j + 3],
			      t_ij5 = t[(i + 1) * n_samples + j], t_ij6 = t[(i + 1) * n_samples + j + 1], t_ij7 = t[(i + 1) * n_samples + j + 2], t_ij8 = t[(i + 1) * n_samples + j + 3],
				  t_ij9 = t[(i + 2) * n_samples + j], t_ij10 = t[(i + 2) * n_samples + j + 1], t_ij11 = t[(i + 2) * n_samples + j + 2], t_ij12 = t[(i + 2) * n_samples + j + 3],
				  t_ij13 = t[(i + 3) * n_samples + j], t_ij14 = t[(i + 3) * n_samples + j + 1], t_ij15 = t[(i + 3) * n_samples + j + 2], t_ij16 = t[(i + 3) * n_samples + j + 3];
			
			float p_ij1 = p[i * n_samples + j], p_ij2 = p[i * n_samples + j + 1], p_ij3 = p[i * n_samples + j + 2], p_ij4 = p[i * n_samples + j + 3],
			      p_ij5 = p[(i + 1) * n_samples + j], p_ij6 = p[(i + 1) * n_samples + j + 1], p_ij7 = p[(i + 1) * n_samples + j + 2], p_ij8 = p[(i + 1) * n_samples + j + 3],
				  p_ij9 = p[(i + 2) * n_samples + j], p_ij10 = p[(i + 2) * n_samples + j + 1], p_ij11 = p[(i + 2) * n_samples + j + 2], p_ij12 = p[(i + 2) * n_samples + j + 3],
				  p_ij13 = p[(i + 3) * n_samples + j], p_ij14 = p[(i + 3) * n_samples + j + 1], p_ij15 = p[(i + 3) * n_samples + j + 2], p_ij16 = p[(i + 3) * n_samples + j + 3];

			float c_ij1 = p_ij1, c_ij2 = p_ij2, c_ij3 = p_ij3, c_ij4 = p_ij4,
			      c_ij5 = p_ij5, c_ij6 = p_ij6, c_ij7 = p_ij7, c_ij8 = p_ij8,
				  c_ij9 = p_ij9, c_ij10 = p_ij10, c_ij11 = p_ij11, c_ij12 = p_ij12,
				  c_ij13 = p_ij13, c_ij14 = p_ij14, c_ij15 = p_ij15, c_ij16 = p_ij16;
			
			c_ij1 -= t_ij1 * t_sum_inv;
			c_ij2 -= t_ij2 * t_sum_inv;
			c_ij3 -= t_ij3 * t_sum_inv;
			c_ij4 -= t_ij4 * t_sum_inv;
			c_ij5 -= t_ij5 * t_sum_inv;
			c_ij6 -= t_ij6 * t_sum_inv;
			c_ij7 -= t_ij7 * t_sum_inv;
			c_ij8 -= t_ij8 * t_sum_inv;
			c_ij9 -= t_ij9 * t_sum_inv;
			c_ij10 -= t_ij10 * t_sum_inv;
			c_ij11 -= t_ij11 * t_sum_inv;
			c_ij12 -= t_ij12 * t_sum_inv;
			c_ij13 -= t_ij13 * t_sum_inv;
			c_ij14 -= t_ij14 * t_sum_inv;
			c_ij15 -= t_ij15 * t_sum_inv;
			c_ij16 -= t_ij16 * t_sum_inv;

			c_ij1 *= t_ij1;
			c_ij2 *= t_ij2;
			c_ij3 *= t_ij3;
			c_ij4 *= t_ij4;
			c_ij5 *= t_ij5;
			c_ij6 *= t_ij6;
			c_ij7 *= t_ij7;
			c_ij8 *= t_ij8;
			c_ij9 *= t_ij9;
			c_ij10 *= t_ij10;
			c_ij11 *= t_ij11;
			c_ij12 *= t_ij12;
			c_ij13 *= t_ij13;
			c_ij14 *= t_ij14;
			c_ij15 *= t_ij15;
			c_ij16 *= t_ij16;

			for (int k = 0; k < d_out; k++) {

				float yik1 = y_trans[k * n_samples + i], yik2 = y_trans[k * n_samples + i + 1], yik3 = y_trans[k * n_samples + i + 2], yik4 = y_trans[k * n_samples + i + 3];
				float yjk1 = y_trans[k * n_samples + j], yjk2 = y_trans[k * n_samples + j + 1], yjk3 = y_trans[k * n_samples + j + 2], yjk4 = y_trans[k * n_samples + j + 3];

				float g_ijk1 = yik1 - yjk1;
				float g_ijk2 = yik1 - yjk2;
				float g_ijk3 = yik1 - yjk3;
				float g_ijk4 = yik1 - yjk4;

				float g_ijk5 = yik2 - yjk1;
				float g_ijk6 = yik2 - yjk2;
				float g_ijk7 = yik2 - yjk3;
				float g_ijk8 = yik2 - yjk4;

				float g_ijk9 = yik3 - yjk1;
				float g_ijk10 = yik3 - yjk2;
				float g_ijk11 = yik3 - yjk3;
				float g_ijk12 = yik3 - yjk4;

				float g_ijk13 = yik4 - yjk1;
				float g_ijk14 = yik4 - yjk2;
				float g_ijk15 = yik4 - yjk3;
				float g_ijk16 = yik4 - yjk4;

				g_ijk1 *= c_ij1;
				g_ijk2 *= c_ij2;
				g_ijk3 *= c_ij3;
				g_ijk4 *= c_ij4;
				g_ijk5 *= c_ij5;
				g_ijk6 *= c_ij6;
				g_ijk7 *= c_ij7;
				g_ijk8 *= c_ij8;
				g_ijk9 *= c_ij9;
				g_ijk10 *= c_ij10;
				g_ijk11 *= c_ij11;
				g_ijk12 *= c_ij12;
				g_ijk13 *= c_ij13;
				g_ijk14 *= c_ij14;
				g_ijk15 *= c_ij15;
				g_ijk16 *= c_ij16;

				g[k * n_samples + i] += g_ijk1 + g_ijk2 + g_ijk3 + g_ijk4;
				//g[k * n_samples + i] += g_ijk2;
				//g[k * n_samples + i] += g_ijk3;
				//g[k * n_samples + i] += g_ijk4;

				g[k * n_samples + i + 1] += g_ijk5 + g_ijk6 + g_ijk7 + g_ijk8;
				//g[k * n_samples + i + 1] += g_ijk6;
				//g[k * n_samples + i + 1] += g_ijk7;
				//g[k * n_samples + i + 1] += g_ijk8;

				g[k * n_samples + i + 2] += g_ijk9 + g_ijk10 + g_ijk11 + g_ijk12;
				//g[k * n_samples + i + 2] += g_ijk10;
				//g[k * n_samples + i + 2] += g_ijk11;
				//g[k * n_samples + i + 2] += g_ijk12;

				g[k * n_samples + i + 3] += g_ijk13 + g_ijk14 + g_ijk15 + g_ijk16;
				//g[k * n_samples + i + 3] += g_ijk14;
				//g[k * n_samples + i + 3] += g_ijk15;
				//g[k * n_samples + i + 3] += g_ijk16;

				g[k * n_samples + j] -= g_ijk1 + g_ijk5 + g_ijk9 + g_ijk13;
				//g[k * n_samples + j] -= g_ijk5;
				//g[k * n_samples + j] -= g_ijk9;
				//g[k * n_samples + j] -= g_ijk13;

				g[k * n_samples + j + 1] -= g_ijk2 +g_ijk6 + g_ijk10 + g_ijk14;
				//g[k * n_samples + j + 1] -= g_ijk6;
				//g[k * n_samples + j + 1] -= g_ijk10;
				//g[k * n_samples + j + 1] -= g_ijk14;

				g[k * n_samples + j + 2] -= g_ijk3 + g_ijk7 + g_ijk11 + g_ijk15;
				//g[k * n_samples + j + 2] -= g_ijk7;
				//g[k * n_samples + j + 2] -= g_ijk11;
				//g[k * n_samples + j + 2] -= g_ijk15;

				g[k * n_samples + j + 3] -= g_ijk4 + g_ijk8 + g_ijk12 + g_ijk16;
				//g[k * n_samples + j + 3] -= g_ijk8;
				//g[k * n_samples + j + 3] -= g_ijk12;
				//g[k * n_samples + j + 3] -= g_ijk16;
			}
		}
	}
}

void gradientUpdate(float *y, float *u, float *g, int n_samples, int d_out, float alpha, float eta) {
	eta *= -4.f;
	for (int i = 0; i < n_samples; i++) {
		for (int k = 0; k < d_out; k++) {
			float u_ik = eta * g[i * d_out + k] + alpha * u[i * d_out + k];
			u[i * d_out + k] = u_ik;
			y[i * d_out + k] += u_ik;
			g[i * d_out + k] = 0.f;
		}
	}
}

void gradientUpdate_trans(float *y_trans, float *u, float *g, int n_samples, int d_out, float alpha, float eta) {
	eta *= -4.f;
	for (int i = 0; i < n_samples; i++) {
		for (int k = 0; k < d_out; k++) {
			float u_ik = eta * g[k * n_samples + i] + alpha * u[k * n_samples + i];
			u[k * n_samples + i] = u_ik;
			y_trans[k * n_samples + i] += u_ik;
			g[k * n_samples + i] = 0.f;
		}
	}
}

/*
void gradientUpdate_trans_block(float *y_trans, float *u, float *g, int n_samples, int d_out, float alpha, float eta) {
	eta *= -4.f;
	for (int i = 0; i < n_samples; i+=4) {
		for (int k = 0; k < d_out; k++) {
			float g_ik1 = g[k * n_samples + i], g_ik2 = g[k * n_samples + i + 1],
			      g_ik3 = g[k * n_samples + i + 2], g_ik4 = g[k * n_samples + i + 3];

			float u_ik01 = u[k * n_samples + i], u_ik02 = u[k * n_samples + i + 1],
			      u_ik03 = u[k * n_samples + i + 2], u_ik04 = u[k * n_samples + i + 3];

			float u_ik1 = eta * g_ik1;
			float u_ik2 = eta * g_ik2;
			float u_ik3 = eta * g_ik3;
			float u_ik4 = eta * g_ik4;
			
			u_ik1 += alpha * u_ik01;
			u_ik2 += alpha * u_ik02;
			u_ik3 += alpha * u_ik03;
			u_ik4 += alpha * u_ik04;

			u[k * n_samples + i] = u_ik1;
			u[k * n_samples + i + 1] = u_ik2;
			u[k * n_samples + i + 2] = u_ik3;
			u[k * n_samples + i + 3] = u_ik4;

			y_trans[k * n_samples + i] += u_ik1;
			y_trans[k * n_samples + i + 1] += u_ik2;
			y_trans[k * n_samples + i + 2] += u_ik3;
			y_trans[k * n_samples + i + 3] += u_ik4;

			g[k * n_samples + i] = 0.f;
			g[k * n_samples + i + 1] = 0.f;
			g[k * n_samples + i + 2] = 0.f;
			g[k * n_samples + i + 3] = 0.f;
		}
	}
}
*/

void gradientUpdate_trans_block(float *y_trans, float *u, float *g, int n_samples, int d_out, float alpha, float eta) {
	eta *= -4.f;
	for (int k = 0; k < d_out; k++) {
	    for (int i = 0; i < n_samples; i+=4) {
			float g_ik1 = g[k * n_samples + i], g_ik2 = g[k * n_samples + i + 1],
			      g_ik3 = g[k * n_samples + i + 2], g_ik4 = g[k * n_samples + i + 3];

			float u_ik01 = u[k * n_samples + i], u_ik02 = u[k * n_samples + i + 1],
			      u_ik03 = u[k * n_samples + i + 2], u_ik04 = u[k * n_samples + i + 3];

			float u_ik1 = eta * g_ik1;
			float u_ik2 = eta * g_ik2;
			float u_ik3 = eta * g_ik3;
			float u_ik4 = eta * g_ik4;
			
			u_ik1 += alpha * u_ik01;
			u_ik2 += alpha * u_ik02;
			u_ik3 += alpha * u_ik03;
			u_ik4 += alpha * u_ik04;

			u[k * n_samples + i] = u_ik1;
			u[k * n_samples + i + 1] = u_ik2;
			u[k * n_samples + i + 2] = u_ik3;
			u[k * n_samples + i + 3] = u_ik4;

			y_trans[k * n_samples + i] += u_ik1;
			y_trans[k * n_samples + i + 1] += u_ik2;
			y_trans[k * n_samples + i + 2] += u_ik3;
			y_trans[k * n_samples + i + 3] += u_ik4;

			g[k * n_samples + i] = 0.f;
			g[k * n_samples + i + 1] = 0.f;
			g[k * n_samples + i + 2] = 0.f;
			g[k * n_samples + i + 3] = 0.f;
		}
	}
}

float gd_pair_aff(float *t, float *y, int n_samples, int d_out) {
	__m256i idx, ones = _mm256_set1_epi32(1);
	__m256  zerofs = _mm256_setzero_ps(), onefs = (__m256) _mm256_set1_epi32(0x3f800000),
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
			} 
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

	return ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7])) + (float) ((N - n_samples) * (N - n_samples) - n_samples);
}

void gd_update_calc(float *u, float *y, float *p, float *t, float t_sum, float eta, int n_samples, int d_out) {
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
					rsum = block_row_sum(a0, a1, a2, a3, a4, a5, a6, a7);
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
		for (int i = 0; i < N; i += 8) {
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
