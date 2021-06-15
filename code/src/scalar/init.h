#ifndef INIT_H
#define INIT_H

#define ALIGN_FLOAT 16

#include "../common/mat.h"

void
init_alloc(float *&d, float *&g, float *&p, float *&t, float *&temp, float *&u, float *&x, float *&y, float *&y_trans,
           int n_samples, int d_in, int d_out) {
	x = mat_alloc<float>(n_samples, d_in);
	p = mat_alloc<float>(n_samples, n_samples);
	temp = mat_alloc<float>(3, n_samples);
	y = mat_alloc<float>(d_out, n_samples);
	t = mat_alloc<float>(n_samples, n_samples);
	u = mat_alloc<float>(d_out, n_samples);
	y_trans = mat_alloc<float>(n_samples, d_out);
}

void init_load(float *x, float *y, float *u, float *g, const std::string &filename, int seed, int n_samples, int d_in,
               int d_out) {
	mat_clear(u, d_out, n_samples);
	mat_clear_margin(y, d_out, n_samples);
	mat_rand_norm(y, d_out, n_samples, 0.f, 1e-4f, true, seed);
	mat_clear_margin(x, n_samples, d_in);
	mat_load(x, n_samples, d_in, filename);
}

#endif //INIT_H
