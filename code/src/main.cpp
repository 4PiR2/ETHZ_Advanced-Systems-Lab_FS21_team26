#include "tsne.h"
//#include "benchmark.h"
#include "tsc_x86.h"
#include "mat.h"
#include "avx512/interface.h"

static void run() {
	run_test();
}

int main(int argc, char *argv[]) {
	// parse parameter

	// run the algorithm
	run();
	return 0;
}
