#ifndef INIT_H
#define INIT_H

#define ALIGN_FLOAT 1

#include "../common/mat.h"

void
init_alloc(float *&d, float *&g, float *&p, float *&t, float *&temp, float *&u, float *&x, float *&y, float *&y_trans,
           int n_samples, int d_in, int d_out) {
	x = mat_alloc<float>(n_samples, d_in);
	d = mat_alloc<float>(n_samples, n_samples);
	p = mat_alloc<float>(n_samples, n_samples);
	y = mat_alloc<float>(n_samples, d_out);
	t = mat_alloc<float>(n_samples, n_samples);
	g = mat_alloc<float>(n_samples, d_out);
	u = mat_alloc<float>(n_samples, d_out);
}

void init_load(float *x, float *y, float *u, float *g, const std::string &filename, int seed, int n_samples, int d_in,
               int d_out) {
	mat_clear(u, n_samples, d_out);
	mat_clear(g, n_samples, d_out);
	mat_rand_norm(y, n_samples, d_out, 0.f, 1e-4f, true, seed);
	mat_load(x, n_samples, d_in, filename);
}

#endif //INIT_H
