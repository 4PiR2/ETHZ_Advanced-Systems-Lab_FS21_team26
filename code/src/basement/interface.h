#ifndef INTERFACE_H
#define INTERFACE_H

#define ALIGNMENT 64

#include "../common/mat.h"
#include "algo.h"

void run(const std::string &file_in, int n_samples, int d_in, int d_out, int n_iter, const std::string &file_out,
         float perplexity, float dummy0, float dummy1, int seed) {
	auto x = mat_alloc<double>(n_samples, d_in), y = mat_alloc<double>(n_samples, d_out);
	mat_load(x, n_samples, d_in, file_in);

	myInt64 s;
	s = start_tsc();
	run_tsne(x, n_samples, d_in, y, d_out, perplexity, seed, n_iter);
	s = stop_tsc(s);

	mat_store(y, n_samples, d_out, file_out);
	mat_free_batch(2, x, y);
	std::cout << s << std::endl;
}

#endif //INTERFACE_H
