#include "tsne.h"
#include "tsc_x86.h"
#include "mat.h"

static void run_baseline(int n_samples, int d_out, int d_in, int rep, float eta, float alpha, float perplexity, int n_iter, std::string file_in, std::string file_out) {

	auto x = mat_alloc<float>(n_samples, d_in),
			p = mat_alloc<float>(n_samples, n_samples),
			p_ex = mat_alloc<float>(n_samples, n_samples),
			t = mat_alloc<float>(n_samples, n_samples),
			y = mat_alloc<float>(d_out, n_samples),
			dy = mat_alloc<float>(d_out, n_samples),
			grad_cy = mat_alloc<float>(d_out, n_samples),
			y_trans = mat_alloc<float>(n_samples, d_out),
			u = mat_alloc<float>(d_out, n_samples),
			temp_3n = mat_alloc<float>(3, n_samples);

	mat_rand_norm<float>(y, n_samples, d_out, 0.f, 0.0001f, true, 17);

	mat_read_data(x, y, u, file_in, 13, n_samples, d_in, d_out);

	getSymmetricAffinity(x, n_samples, d_in, perplexity, p);

	getLowDimResult(y, dy, grad_cy, p, t, n_samples, d_out, alpha, eta, n_iter);
	
	mat_store(y, n_samples, d_out, file_out);

	mat_free_batch(8, x, p, p_ex, t, y, y_trans, u, temp_3n);
}


int main(int argc, char *argv[]) {
	int n_samples = 900, d_in = 784, d_out = 3, n_iter = 1000, n_iter_ex = 250, seed = 0;
	float perplexity = 50.f, ex_rate = 12.f, eta = 50.f, alpha = .8f, alpha_ex = .5f;

	int rep = 3;

	std::string file_in = "code/datasets/mnist/mnist_data_70kx784.txt", file_out = "code/output_matrix.txt";

	run_baseline(n_samples, d_out, d_in, rep, eta, alpha, perplexity, n_iter, file_in, file_out);

	return 0;
}