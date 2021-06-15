#include "common/tsc_x86.h"

#define V_SCALAR
// Warning: AVX2 is currently not available!

#ifdef V_BASEMENT

#include "basement/interface.h"

#endif
#ifdef V_BASELINE

#include "baseline/init.h"
#include "baseline/pre.h"
#include "baseline/gd.h"

#endif
#ifdef V_SCALAR

#include "scalar/init.h"
#include "scalar/pre.h"
#include "scalar/gd.h"

#endif
#ifdef V_AVX2

#include "baseline/init.h"
#include "baseline/pre.h"
#include "avx2/sym_aff.h"
#include "avx2/gd.h"

#endif
#ifdef V_AVX512

#include "scalar/init.h"
#include "avx512/pre.h"
#include "avx512/gd.h"

#endif


#ifndef V_BASEMENT
void run(const std::string &file_in, int n_samples, int d_in, int d_out, int n_iter, const std::string &file_out,
		 float perplexity, float eta, float alpha, int seed) {

	myInt64 s, ta0 = 0, ta1 = 0, ta2 = 0, ta3 = 0, tb0 = 0, tb1 = 0, tb2 = 0;
	float *d = nullptr, *g = nullptr, *p = nullptr, *t = nullptr, *temp = nullptr, *u = nullptr, *x = nullptr, *y = nullptr, *y_trans = nullptr;
	int count;

	init_alloc(d, g, p, t, temp, u, x, y, y_trans, n_samples, d_in, d_out);
	init_load(x, y, u, g, file_in, seed, n_samples, d_in, d_out);

	s = start_tsc();
	pre_pair_sq_dist(p, d, x, temp, n_samples, d_in);
	ta0 += stop_tsc(s);
	s = start_tsc();
	pre_unfold_low_tri(p, n_samples);
	ta1 += stop_tsc(s);
	//mat_store(p, n_samples, n_samples, "../output/d_mut.txt");
	s = start_tsc();
	count = pre_perplex_bi_search(p, d, perplexity, 1e-5f, temp, n_samples);
	ta2 += stop_tsc(s);
	//mat_store(p, n_samples, n_samples, "../output/p_mut.txt");
	s = start_tsc();
	pre_sym_aff(p, n_samples);
	ta3 += stop_tsc(s);
	//pre_unfold_low_tri(p, n_samples);
	//mat_store(p, n_samples, n_samples, "../output/ps_mut.txt");

	float t_sum_inv;
	for (int i = 0; i < n_iter; ++i) {
		s = start_tsc();
		t_sum_inv = gd_pair_aff(t, y, n_samples, d_out);
		tb0 += stop_tsc(s);
		s = start_tsc();
		gd_update_calc(u, g, y, p, t, t_sum_inv, n_samples, d_out);
		tb1 += stop_tsc(s);
		s = start_tsc();
		gd_update_apply(y, u, g, eta * 4.f, alpha, n_samples, d_out);
		tb2 += stop_tsc(s);
	}

	if (y_trans) {
		mat_transpose(y_trans, y, d_out, n_samples);
		mat_store(y_trans, n_samples, d_out, file_out);
	} else {
		mat_store(y, n_samples, d_out, file_out);
	}

	mat_free_batch(9, d, g, p, t, temp, u, x, y, y_trans);
	std::cout << ta0 << "\t" << ta1 << "\t" << ta2 << "\t" << ta3 << "\t" << tb0 << "\t" << tb1 << "\t" << tb2 << "\n"
			  << count << std::endl;
}
#endif


int main(int argc, char *argv[]) {
	std::string file_in, file_out = "/dev/null";
	int n_samples, d_in = 784, d_out = 2, n_iter = 1000, seed = 0;
	float perplexity = 50.f, eta = 50.f, alpha = .8f;
	switch (argc) {
		case 11:
			seed = atoi(argv[10]);
		case 10:
			alpha = atof(argv[9]);
		case 9:
			eta = atof(argv[8]);
		case 8:
			perplexity = atof(argv[7]);
		case 7:
			file_out = argv[6];
		case 6:
			n_iter = atoi(argv[5]);
		case 5:
			d_out = atoi(argv[4]);
		case 4:
			d_in = atoi(argv[3]);
		case 3:
			n_samples = atoi(argv[2]);
			file_in = argv[1];
			break;
		default:
			std::cerr
					<< "string file_in, int n_samples, [int d_in, int d_out, int n_iter, string file_out, float perplexity, float eta, float alpha, int seed]"
					<< std::endl;
	}

	int test = 0;
	if (test == 0) {
		run(file_in, n_samples, d_in, d_out, n_iter, file_out, perplexity, eta, alpha, seed);
	} /* else if (test == 1) {
		n_samples = 100;
		d_out = 3;
		n_iter = 3;

		auto p = mat_alloc<float>(n_samples, n_samples),
				t = mat_alloc<float>(n_samples, n_samples),
				y = mat_alloc<float>(d_out, n_samples),
				y_trans = mat_alloc<float>(n_samples, d_out),
				u = mat_alloc<float>(d_out, n_samples);

		mat_clear(u, d_out, n_samples);
		mat_clear_margin(y, d_out, n_samples);
		mat_load(p, n_samples, n_samples, "../datasets/random/random_sym_1000x1000.txt");
		mat_load(y, d_out, n_samples, "../datasets/random/random_normal_10x10000.txt");
		mat_load(y, d_out, n_samples, "../datasets/random/random_sym_1000x1000.txt");

		float t_sum_inv;
		for (int i = 0; i < n_iter; ++i) {
			//t_sum_inv = gd_pair_aff(t, y, n_samples, d_out);
			//mat_store(t, n_samples, n_samples, "../output/t_mut.txt");
			//gd_update_calc(u, nullptr, y, p, t, t_sum_inv, n_samples, d_out);
			//gd_update_apply(y, u, nullptr, eta * 4.f, alpha, n_samples, d_out);
		}

		mat_transpose(y_trans, y, d_out, n_samples);
		mat_store(y_trans, n_samples, d_out, "../output/y_mut.txt");

		mat_free_batch(5, p, t, y, y_trans, u);
	} else if (test == 2) {
		myInt64 s, ta0 = 0, ta1 = 0, ta2 = 0, ta3 = 0, tb0 = 0, tb1 = 0, tb2 = 0;
		float *d = nullptr, *g = nullptr, *p = nullptr, *t = nullptr, *temp = nullptr, *u = nullptr, *x = nullptr, *y = nullptr, *y_trans = nullptr;
		int count;

		init_alloc(d, g, p, t, temp, u, x, y, y_trans, n_samples, d_in, d_out);
		init_load(x, y, u, g, file_in, seed, n_samples, d_in, d_out);

		s = start_tsc();
		pre_pair_sq_dist(p, d, x, temp, n_samples, d_in);
		ta0 += stop_tsc(s);
		s = start_tsc();
		pre_unfold_low_tri(p, n_samples);
		ta1 += stop_tsc(s);
		//mat_store(p, n_samples, n_samples, "../output/d_mut.txt");
		s = start_tsc();
		//getPairwiseAffinity(d, n_samples, perplexity, p);
		pre_perplex_bi_search(p, d, perplexity, 1e-5f, temp, n_samples);
		ta2 += stop_tsc(s);
		//mat_store(p, n_samples, n_samples, "../output/p_mut.txt");
		s = start_tsc();
		pre_sym_aff(p, n_samples);
		ta3 += stop_tsc(s);
		//pre_unfold_low_tri(p, n_samples);
		//mat_store(p, n_samples, n_samples, "../output/ps_mut.txt");

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

		if (y_trans) {
			mat_transpose(y_trans, y, d_out, n_samples);
			mat_store(y_trans, n_samples, d_out, file_out);
		} else {
			mat_store(y, n_samples, d_out, file_out);
		}

		mat_free_batch(9, d, g, p, t, temp, u, x, y, y_trans);
		std::cout << ta0 << "\t" << ta1 << "\t" << ta2 << "\t" << ta3 << "\t" << tb0 << "\t" << tb1 << "\t" << tb2 << "\n"
		          << count << std::endl;
	} */
	return 0;
}
