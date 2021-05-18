#include <assert.h>

#include "baselines.h"
#include "math.h"

#define DEBUG

#define eps_baselines 1e-7
// C is the assertion message
#define assertEq(A,B,C) assert(fabs(A - B) <= eps_baselines && C) 
/*
    the output of the original function is stored in DD argument
    we need to re-generate the whole array and make comparisons

    DD has a size of n_samples * n_samples
*/
void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD_output) {
#ifdef DEBUG
    float* DD = mat_alloc_float(n_samples, n_samples);
    const float* XnD = X;
    for(int n = 0; n < n_samples; ++n, XnD += dim) {
        const float* XmD = XnD + dim;
        float* curr_elem = &DD[n*n_samples + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + n_samples;
        for(int m = n + 1; m < n_samples; ++m, XmD+=dim, curr_elem_sym+=n_samples) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < dim; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_samples; j++) {
            int idx = i * n_samples + j;
            assertEq(DD[idx], DD_output[idx], "ED TEST FAILS");
        }
    }

    mat_free_float(DD);
#endif
}