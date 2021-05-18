#include <stdio.h>

void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD) {
    // const float* XnD = X;
    // for(int n = 0; n < n_samples; ++n, XnD += dim) {
    //     const float* XmD = XnD + dim;
    //     float* curr_elem = &DD[n*n_samples + n];
    //     *curr_elem = 0.0;
    //     float* curr_elem_sym = curr_elem + n_samples;
    //     for(int m = n + 1; m < n_samples; ++m, XmD+=dim, curr_elem_sym+=n_samples) {
    //         *(++curr_elem) = 0.0;
    //         for(int d = 0; d < dim; ++d) {
    //             *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
    //         }
    //         *curr_elem_sym = *curr_elem;
    //     }
    // }

    for (int i = 0; i < n_samples; i++) {
        for (int j = i + 1; j < n_samples; j++) {
            float tmp = 0.0;
            for (int k = 0; k < dim; k++) {
                float sq = X[i * dim + k] - X[j * dim + k];
                tmp += sq * sq;
            }
            DD[i * n_samples + j] = DD[j * n_samples + i] = tmp;
        }
        DD[i * n_samples + i] = 0.0;
    }
}

static void testGetSquaredEuclideanDistances(void) {
    float x[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int n = 3;
    int d = 3;
    float res[9];

    _getSquaredEuclideanDistances((float*)x, n, d, (float*)res);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            printf("%.2lf ", res[idx]);
        }
        printf("\n");
    }
}

int main() {
    testGetSquaredEuclideanDistances();
}