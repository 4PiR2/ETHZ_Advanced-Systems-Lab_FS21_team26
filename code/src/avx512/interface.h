#ifndef INTERFACE_H
#define INTERFACE_H

#include "mat.h"
#include "pre.h"
#include "gd.h"

void readData(float *x, float *y, float *u, const std::string &filename, int seed, int n_samples, int d_in, int d_out) {
	mat_clear(u, d_out, n_samples);
	mat_clear_margin(y, d_out, n_samples);
	mat_rand_norm(y, d_out, n_samples, 0.f, 1e-4f, true, seed);
	mat_clear_margin(x, n_samples, d_in);
	mat_load(x, n_samples, d_in, filename);
}

void getSymmetricAffinity(float *p, float *p_ex, float *x, float *temp_3n, float perplexity, float ex_rate,
                          int n_samples, int d_in) {
	pre_pair_sq_dist(p, x, n_samples, d_in);
	pre_unfold_low_tri(p, n_samples);
	pre_perplex_bi_search(p, perplexity, 1e-5f, 200, temp_3n, n_samples);
	pre_sym_aff_ex(p, p_ex, ex_rate, n_samples);
}

void getLowdResult(float *y, float *u, float *p, float *p_ex, float *t,
                   float eta, int n_iter, int n_iter_ex, float alpha, float alpha_ex, int n_samples, int d_out) {
	float t_sum;
	for (int i = 0; i < n_iter; ++i) {
		t_sum = gd_pair_aff(t, y, n_samples, d_out);
		if (i < n_iter_ex) {
			gd_update_calc(u, y, p_ex, t, t_sum, eta, n_samples, d_out);
			gd_update_apply(y, u, alpha_ex, n_samples, d_out);
		} else {
			gd_update_calc(u, y, p, t, t_sum, eta, n_samples, d_out);
			gd_update_apply(y, u, alpha, n_samples, d_out);
		}
	}
}

void run_test() {
	int n_samples = 900, d_in = 784, d_out = 3, n_iter = 1000, n_iter_ex = 250, seed = 0;
	float perplexity = 50.f, ex_rate = 12.f, eta = 50.f, alpha = .8f, alpha_ex = .5f;
	std::string file_in = "datasets/mnist/mnist_data_70kx784.txt", file_out = "output_matrix.txt";

	auto x = mat_alloc<float>(n_samples, d_in),
			p = mat_alloc<float>(n_samples, n_samples),
			p_ex = mat_alloc<float>(n_samples, n_samples),
			t = mat_alloc<float>(n_samples, n_samples),
			y = mat_alloc<float>(d_out, n_samples),
			y_trans = mat_alloc<float>(n_samples, d_out),
			u = mat_alloc<float>(d_out, n_samples),
			temp_3n = mat_alloc<float>(3, n_samples);

	readData(x, y, u, file_in, seed, n_samples, d_in, d_out);

	getSymmetricAffinity(p, p_ex, x, temp_3n, perplexity, ex_rate, n_samples, d_in);

	getLowdResult(y, u, p, p_ex, t, eta, n_iter, n_iter_ex, alpha, alpha_ex, n_samples, d_out);

	mat_transpose(y_trans, y, d_out, n_samples);
	mat_store(y_trans, n_samples, d_out, file_out);

	mat_free_batch(8, x, p, p_ex, t, y, y_trans, u, temp_3n);
}

#endif //INTERFACE_H
