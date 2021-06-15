#ifndef GD_H
#define GD_H

float gd_pair_aff(float *t, float *y, int n_samples, int d_out) {
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

void gd_update_calc(float *dummy0, float *g, float *y, float *p, float *t, float t_sum_inv, int n_samples, int d_out) {
	for (int i = 0; i < n_samples; i++) {
		for (int j = 0; j < n_samples; j++) {
			float t_ij = t[i * n_samples + j], c_ij = (p[i * n_samples + j] - t_ij * t_sum_inv) * t_ij;
			for (int k = 0; k < d_out; k++) {
				g[i * d_out + k] += (y[i * d_out + k] - y[j * d_out + k]) * c_ij;
			}
		}
	}
}

void gd_update_apply(float *y, float *u, float *g, float etax4, float alpha, int n_samples, int d_out) {
	for (int i = 0; i < n_samples; i++) {
		for (int k = 0; k < d_out; k++) {
			float u_ik = alpha * u[i * d_out + k] - etax4 * g[i * d_out + k];
			u[i * d_out + k] = u_ik;
			y[i * d_out + k] += u_ik;
			g[i * d_out + k] = 0.f;
		}
	}
}

#endif //GD_H
