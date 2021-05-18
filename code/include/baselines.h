#ifndef BASELINES_H
#define BASELINES_H

#include <stdlib.h>
#include "mat.h"

void _getPairwiseAffinity(const float* ED, int n_samples, float perp, float* P);
void _normalizeData(float* X, int n_samples, int d_in);
void _symmetrizeAffinities(float* P, int n_samples);
void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD);
void baselineCompare(const float* X, const float* Y, int size, const char* msg);
#endif