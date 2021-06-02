#ifndef INTERFACE_H
#define INTERFACE_H

#define ALIGN_ELEM 16

#include "../common/tsc_x86.h"
#include "../common/mat.h"
#include "pre.h"
#include "gd.h"

myInt64 s, ta0 = 0, ta1 = 0, ta2 = 0, ta3 = 0, tb0 = 0, tb1 = 0, tb2 = 0;

void readData(float *x, float *y, float *u, const std::string &filename, int seed, int n_samples, int d_in, int d_out) {
	mat_clear(u, d_out, n_samples);
	mat_clear_margin(y, d_out, n_samples);
	mat_rand_norm(y, d_out, n_samples, 0.f, 1e-4f, true, seed);
	mat_clear_margin(x, n_samples, d_in);
	mat_load(x, n_samples, d_in, filename);
}

int getSymmetricAffinity(float *p, float *x, float *temp_3n, float perplexity, int n_samples, int d_in) {
	s = start_tsc();
	pre_pair_sq_dist_2x8(p, x, temp_3n, n_samples, d_in);
	ta0 += stop_tsc(s);
	s = start_tsc();
	pre_unfold_low_tri(p, n_samples);
	ta1 += stop_tsc(s);
	s = start_tsc();
	int count = pre_perplex_bi_search(p, perplexity, 1e-5f, temp_3n, n_samples);
	ta2 += stop_tsc(s);
	s = start_tsc();
	pre_sym_aff(p, n_samples);
	ta3 += stop_tsc(s);
	return count;
}

void
getLowdResult(float *y, float *u, float *p, float *t, float eta, int n_iter, float alpha, int n_samples, int d_out) {
	float t_sum;
	for (int i = 0; i < n_iter; ++i) {
		s = start_tsc();
		t_sum = gd_pair_aff(t, y, n_samples, d_out);
		tb0 += stop_tsc(s);
		s = start_tsc();
		gd_update_calc(u, y, p, t, t_sum, eta, n_samples, d_out);
		tb1 += stop_tsc(s);
		s = start_tsc();
		gd_update_apply(y, u, alpha, n_samples, d_out);
		tb2 += stop_tsc(s);
	}
}

void run(int n_samples, int d_out, int d_in, int rep, float eta, float alpha, float perplexity, int n_iter,
         const std::string &file_in, const std::string &file_out) {
	int seed = 0;

	auto x = mat_alloc<float>(n_samples, d_in),
			p = mat_alloc<float>(n_samples, n_samples),
			t = mat_alloc<float>(n_samples, n_samples),
			y = mat_alloc<float>(d_out, n_samples),
			y_trans = mat_alloc<float>(n_samples, d_out),
			u = mat_alloc<float>(d_out, n_samples),
			temp_3n = mat_alloc<float>(3, n_samples);

	readData(x, y, u, file_in, seed, n_samples, d_in, d_out);

	int count = getSymmetricAffinity(p, x, temp_3n, perplexity, n_samples, d_in);

	getLowdResult(y, u, p, t, eta, n_iter, alpha, n_samples, d_out);

	mat_transpose(y_trans, y, d_out, n_samples);
	mat_store(y_trans, n_samples, d_out, file_out);

	mat_free_batch(8, x, p, t, y, y_trans, u, temp_3n);
	std::cout << ta0 << "\t" << ta1 << "\t" << ta2 << "\t" << ta3 << "\t" << tb0 << "\t" << tb1 << "\t" << tb2 << "\n"
	          << count << std::endl;
}

#endif //INTERFACE_H
