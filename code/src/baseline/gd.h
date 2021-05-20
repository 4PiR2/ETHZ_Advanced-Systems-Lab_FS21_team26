#ifndef GD_H
#define GD_H

void compute_t(float *y, float *t, int n, int d) {
	for (int i = 0; i < n; i++) {
		t[i * n + i] = 0.f;
		for (int j = i + 1; j < n; j++) {
			float dist = 1.f;
			for (int k = 0; k < d; k++) {
				float diff = y[i * d + k] - y[j * d + k];
				dist += diff * diff;
			}
			t[j * n + i] = t[i * n + j] = 1.f / dist;
		}
	}
}

float compute_sum_t_inv(float *t, int n) {
	float sum_t = 0.f;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			sum_t += t[i * n + j];
		}
	}
	return .5f / sum_t;
}

void gradientCompute(float *y, float *g, float *p, float *t, float sum_t_inv, int n, int d) {
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			float t_ij = t[i * n + j], c_ij = (p[i * n + j] - t_ij * sum_t_inv) * t_ij;
			for (int k = 0; k < d; k++) {
				float g_ijk = (y[i * d + k] - y[j * d + k]) * c_ij;
				g[i * d + k] += g_ijk;
				g[j * d + k] -= g_ijk;
			}
		}
	}
}

void gradientUpdate(float *y, float *u, float *g, int n, int d, float alpha, float eta) {
	eta *= -4.f;
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < d; k++) {
			float u_ik = eta * g[i * d + k] + alpha * u[i * d + k];
			u[i * d + k] = u_ik;
			y[i * d + k] += u_ik;
			g[i * d + k] = 0.f;
		}
	}
}

#endif //GD_H
