#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

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

void gradientCompute(float *y, float *grad_cy, float *p, float *t, float sum_t_inv, int n, int d) {
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			float c = (p[i * n + j] - t[i * n + j] * sum_t_inv) * t[i * n + j];
			for (int k = 0; k < d; k++) {
				float g = (y[i * d + k] - y[j * d + k]) * c;
				grad_cy[i * d + k] += g;
				grad_cy[j * d + k] -= g;
			}
		}
	}
}

void gradientUpdate(float *y, float *u, float *grad_cy, int n, int d, float alpha, float eta) {
	eta *= -4.f;
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < d; k++) {
			u[i * d + k] = eta * grad_cy[i * d + k] + alpha * u[i * d + k];
			grad_cy[i * d + k] = 0.f;
			y[i * d + k] += u[i * d + k];
		}
	}
}

#endif //GRADIENT_DESCENT_H
