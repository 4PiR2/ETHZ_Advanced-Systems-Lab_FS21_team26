#include "tsne.h"
#include "benchmark.h"
#include "tsc_x86.h"
#include "mat.h"
#include "avx512/low_dim.h"


static void run() {
	int n_samples = 900, d_out = 3, rep = 3;
	float eta = 50.f, alpha = .5f;

	// read data (sth in mat.h)

	auto y = mat_alloc<float>(d_out, n_samples), t = mat_alloc<float>(n_samples, n_samples),
			u = mat_alloc<float>(d_out, n_samples), p = mat_alloc<float>(n_samples, n_samples),
			y_trans = mat_alloc<float>(n_samples, d_out);
	mat_load(p, n_samples, n_samples,
	         "datasets/random/random_sym_1000x1000.txt");
	mat_clear(u, d_out, n_samples);
	mat_load(y, d_out, n_samples,
	         "datasets/random/random_normal_10x10000.txt");
	mat_clear_margin(y, d_out, n_samples);

	// get symmetric affinity
	// getSymmetricAffinity(a, 1, 1, 1, b);

	// gradient descent algorithm
	// getLowdResult(a, 1, 1, b);

	myInt64 start, t1 = 0, t2 = 0, t3 = 0;

	float t_sum;
	for (int i = 0; i < rep; ++i) {
		start = start_tsc();
		t_sum = low_dim_pair_aff(t, y, n_samples, d_out);
		t1 += stop_tsc(start);
		start = start_tsc();
		low_dim_update_calc(u, y, p, t, t_sum, eta, n_samples, d_out);
		t2 += stop_tsc(start);
		start = start_tsc();
		low_dim_update_apply(y, u, alpha, n_samples, d_out);
		t3 += stop_tsc(start);
		// the normalizing part should not be in the algorithm
	}

	t1 /= rep;
	t2 /= rep;
	t3 /= rep;
	std::cout << t1 << "\t" << t2 << "\t" << t3 << std::endl;
	std::cout << t1 + t2 + t3 << std::endl;

	// write back the final result

	mat_transpose(y_trans, y, d_out, n_samples);
	mat_store(y_trans, n_samples, d_out, "output_matrix.txt");
	mat_free(y);
	mat_free(t);
	mat_free(u);
	mat_free(p);
	mat_free(y_trans);
}

int main(int argc, char *argv[]) {
	// parse parameter

	// run the algorithm
	run();
	return 0;
}
