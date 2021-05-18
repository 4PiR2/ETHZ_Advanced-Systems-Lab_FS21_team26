#ifndef BASELINES_H
#define BASELINES_H

#include <stdlib.h>
#include "mat.h"

void _getPairwiseAffinity(float* X, int n_samples, int dim, float* DD);
void _normalizeData(float* X, int n_samples, int d_in);
void _symmetrizeAffinities(float* P, int n_samples);
void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD);
void baselineCompare(const float* X, const float* Y, int size, char* msg);
#endif