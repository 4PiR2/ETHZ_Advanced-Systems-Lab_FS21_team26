#ifndef BASELINES_H
#define BASELINES_H

#include <stdlib.h>

void _getPairwiseAffinity(float* X, int n_samples, int dim, float* DD);
void _normalizeData(float* X, int n_samples, int d_in);
void _symmetrizeAffinities(float* P, int n_samples);
void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD);

/*
    Since Mat.h is implemented in template, it is hard to include it in different
    c files within the separate compilation model
*/
#define ALIGNMENT 32
#define GET_N(n, T) (((n) + (((ALIGNMENT) >> 3) / sizeof(T) - 1)) & (-1 ^ (((ALIGNMENT) >> 3) / sizeof(T) - 1)))
inline
float *mat_alloc_float(int n, int m) {
	float *p = (float*)malloc(n * m * sizeof(float));
    return p;
}

inline
void mat_free_float(float* p) {
    free(p);
}

#endif