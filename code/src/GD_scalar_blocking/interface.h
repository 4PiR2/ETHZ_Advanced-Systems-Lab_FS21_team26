#ifndef INTERFACE_H
#define INTERFACE_H

#define ALIGN_ELEM 1

#include "../common/mat.h"
#include "pre.h"
#include "gd.h"

void readData(float *x, float *y, float *y_trans, const std::string &filename, int seed, int n_samples, int d_in, int d_out) {
	mat_rand_norm(y, n_samples, d_out, 0.f, 1e-4f, true, seed);
	mat_transpose(y_trans, y, n_samples, d_out);
	mat_load(x, n_samples, d_in, filename);
}

void getSymmetricAffinity(float *x, int n_samples, int d_in, float perplexity, float *p, float *d) {
	_getSquaredEuclideanDistances(x, n_samples, d_in, d);
	_getPairwiseAffinity(d, n_samples, perplexity, p);
	_symmetrizeAffinities(p, n_samples);
}

void getLowDimResult(float *y, float *y_trans, float *u, float *g, float *p, float *t, int n_samples, int d_out, float alpha,
                     float eta, int n_iter) {
	for (int i = 0; i < n_iter; i++) {
		float t_sum_inv = compute_t_trans_block(y_trans, t, n_samples, d_out);//wrong
		gradientCompute_trans_block(y_trans, g, p, t, t_sum_inv, n_samples, d_out);//wrong
		gradientUpdate_trans_block(y_trans, u, g, n_samples, d_out, alpha, eta);//wrong
	}
}

static void
run(int n_samples, int d_out, int d_in, int rep, float eta, float alpha, float perplexity, int n_iter,
    const std::string &file_in, const std::string &file_out) {
	thandle t1 = create_timer("SymAff"), t2 = create_timer("GradDesc");
	for (int r = 0; r < rep; r++) {
		printf("Rep %d starting...\n", r + 1);
		auto x = mat_alloc<float>(n_samples, d_in),
				p = mat_alloc<float>(n_samples, n_samples),
				t = mat_alloc<float>(n_samples, n_samples),
				y = mat_alloc<float>(n_samples, d_out),
				y_trans = mat_alloc<float>(d_out, n_samples),
				u = mat_alloc<float>(n_samples, d_out),
				g = mat_alloc<float>(n_samples, d_out),
				d = mat_alloc<float>(n_samples, n_samples);
		readData(x, y, y_trans, file_in, 13, n_samples, d_in, d_out);

		start(t1);
		getSymmetricAffinity(x, n_samples, d_in, (float) perplexity, p, d);
		stop(t1);

		mat_store(p, n_samples, n_samples, "../output/p_matrix.txt");

		start(t2);
		getLowDimResult(y, y_trans, u, g, p, t, n_samples, d_out, alpha, eta, n_iter);
		stop(t2);

		mat_transpose(y, y_trans, d_out, n_samples);

		mat_store(y, n_samples, d_out, file_out);

		mat_free_batch(7, x, p, t, y, u, g, d);
	}

	benchmark_print();
}

#endif //INTERFACE_H