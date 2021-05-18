#include <assert.h>

#include "baselines.h"
#include "math.h"

#define DEBUG

#define eps_baselines 1e-7
// C is the assertion message
#define assertEq(A,B,C) assert(fabs(A - B) <= eps_baselines && C) 

void baselineCompare(const float* X, const float* Y, const int size, char* msg) {
#ifdef DEBUG
    for (int i = 0; i < size; i++)
        assertEq(X[i], Y[i], msg);
#endif
}

void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD) {
#ifdef DEBUG
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
#endif
}