#include <stdio.h>
#include <math.h>
#include <limits>
#include "../include/tsne.h"

// define the maximum number of iterations to fit the perplexity
int MAX_ITERATIONS = 1000;

// define the error tolerance for the perplexity
double ERROR_TOLERANCE = 1e-5;

double MIN_DOUBLE = std::numeric_limits<double>::min();
double MAX_DOUBLE = std::numeric_limits<double>::max();

void getPairwiseAffinity(double* squaredEuclidianDistances, int n, int perplexity, double* affinity) {
    double log_perp = log(perplexity);

    // compute affinities row by row
    for (int i = 0; i < n*n; i += n) {
        // initialize beta values
        double beta = 1.0;
        double beta_max = MAX_DOUBLE;
        double beta_min = MIN_DOUBLE;

        double sum;
        // perform binary search to find the optimal beta values for each data point
        for (int k = 0; k < MAX_ITERATIONS; ++k) {

            // compute the conditional Gaussian densities for point i
            sum = 0.0;
			for(int j = 0; j < n; j++) {
                double gaussian_density = exp(-beta * squaredEuclidianDistances[n*i + j]);
			    affinity[n*i + j] = gaussian_density;
                sum += gaussian_density;
			}

			double shannon_entropy = 0.0;
			for (int j = 0; j < n; j++) shannon_entropy += beta * (squaredEuclidianDistances[n*i + j] * affinity[n*i + j]);
			shannon_entropy = (shannon_entropy / sum) + log(sum);

			double entropy_error = shannon_entropy - log_perp;
			if (abs(entropy_error) < ERROR_TOLERANCE) {
                break;
			} else {
				if(entropy_error > 0) {
					beta_min = beta;
					if(abs(beta_max) == MAX_DOUBLE)
						beta *= 2.0;
					else
						beta = (beta + beta_max) / 2.0;
				}
				else {
					beta_max = beta;
					if(abs(beta_min) == MIN_DOUBLE)
						beta /= 2.0;
					else
						beta = (beta + beta_min) / 2.0;
				}
			}
        }

        // normalize the row
        for(int j = 0; j < n; j++) {
            affinity[n*i + j] /= sum;
        }
    }
}

void getSymmetricAffinity(double* x, int n, int d, int perp, double* affinity) {
    double* squaredEuclidianDistances = (double*) malloc(n * n * sizeof(double));

    if (squaredEuclidianDistances == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
	getSquaredEuclideanDistances(x, n, d, squaredEuclidianDistances);

    // compute pairwise affinities
    getPairwiseAffinity(squaredEuclidianDistances, n, perp, affinity);

    free(squaredEuclidianDistances);

    // compute symmetric affinities
    double normalization_factor = 1/(2*n);
    for(int i = 0; i < n*n; i += n) {
        for(int j = i+1; j < n; j++) {
            double p_ij = affinity[n*i + j];
            double p_ji = affinity[n*j + i];

            double symmetric_affinity = (p_ij + p_ji) * normalization_factor;
            affinity[n*i + j] = symmetric_affinity;
            affinity[n*j + i] = symmetric_affinity;
        }
    }
}

// Compute squared Euclidean distance matrix
void getSquaredEuclideanDistances(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}