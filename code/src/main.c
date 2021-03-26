#include "tsne.h"
#include "io.h"
#include "benchmark.h"

static void run() {
    // read data (sth in io.h)

    double *a, *b;
    // get symmetric affinity
    getSymmetricAffinity(a, 1, 1, 1, b);
    // gradient descent algorithm
    getLowdResult(a, 1, 1, b);
    // write back the final result
}

int main(int argc, char* argv[]) {
    // parse parameter

    // run the algorithm
    run();
    return 0;
}