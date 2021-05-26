#include "common/benchmark.h"

// #define V_SCALAR
// #define V_BASELINE

#ifdef V_BASELINE

#include "baseline/interface.h"

#endif

#ifdef V_SCALAR

#include "scalar/interface.h"

#endif

int ntimers = 0;
struct timer timers[MAXN_TIMERS];

// TODO: very ugly hard coding here
char timers_to_show[][MAX_TIMER_NAME_LEN] = {
		"SymAff",
		"GradDesc",
		"ED",
		"_ED",
		"PA",
		"_PA",
		"SA",
		"_SA",
};
int n_timers_to_show = sizeof(timers_to_show) / MAX_TIMER_NAME_LEN;
// first: current, second: baseline
char timers_to_compare[][2][MAX_TIMER_NAME_LEN] = {
		// {"ED", "_ED"},
		{"PA", "_PA"},
		// {"SA", "_SA"},
};
int n_timers_to_compare = sizeof(timers_to_compare) / (2 * MAX_TIMER_NAME_LEN);


int main(int argc, char *argv[]) {
	int n_samples, d_in = 784, d_out = 2, n_iter = 1000 /*,n_iter_ex = 250*/ /*,seed = 0*/;
	float perplexity = 50.f /*,ex_rate = 12.f*/, eta = 50.f, alpha = .1f/*,alpha_ex = .5f*/;

	n_samples = atoi(argv[1]);
	
	int rep = 5;

	std::string file_in = "../datasets/mnist/mnist_data_70kx784.txt", file_out = "../output/output_matrix.txt";

	run(n_samples, d_out, d_in, rep, eta, alpha, perplexity, n_iter, file_in, file_out);

	return 0;
}