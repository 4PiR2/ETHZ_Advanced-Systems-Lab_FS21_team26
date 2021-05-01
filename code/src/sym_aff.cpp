#include <stdio.h>
#include <math.h>
#include <limits>
#include "tsne.h"

// define the maximum number of iterations to fit the perplexity
int MAX_ITERATIONS = 1000;

// define the error tolerance for the perplexity
float ERROR_TOLERANCE = 1e-5;

float MIN_FLOAT = std::numeric_limits<float>::min();
float MAX_FLOAT = std::numeric_limits<float>::max();

void getPairwiseAffinity(float* squaredEuclidianDistances, int n, int perplexity, float* affinity) {
    float log_perp = log(perplexity);

    // compute affinities row by row
    for (int i = 0; i < n; i++) {
        // initialize beta values
        float beta = 1.0;
        float beta_max = MAX_FLOAT;
        float beta_min = MIN_FLOAT;

        float sum;
        // perform binary search to find the optimal beta values for each data point
        for (int k = 0; k < MAX_ITERATIONS; ++k) {

            // compute the conditional Gaussian densities for point i
            sum = 0.0;
			for(int j = 0; j < n; j++) {
                float gaussian_density = exp(-beta * squaredEuclidianDistances[n*i + j]);
			    affinity[n*i + j] = gaussian_density;
                sum += gaussian_density;
			}

			float shannon_entropy = 0.0;
			for (int j = 0; j < n; j++) shannon_entropy += beta * (squaredEuclidianDistances[n*i + j] * affinity[n*i + j]);
			shannon_entropy = (shannon_entropy / sum) + log(sum);

			float entropy_error = shannon_entropy - log_perp;
			if (abs(entropy_error) < ERROR_TOLERANCE) {
                break;
			} else {
				if(entropy_error > 0) {
					beta_min = beta;
					if(abs(beta_max) == MAX_FLOAT)
						beta *= 2.0;
					else
						beta = (beta + beta_max) / 2.0;
				}
				else {
					beta_max = beta;
					if(abs(beta_min) == MIN_FLOAT)
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

void getSymmetricAffinity(float* x, int n, int d, int perp, float* affinity) {
    float* squaredEuclidianDistances = (float*) malloc(n * n * sizeof(float));

    if (squaredEuclidianDistances == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
	getSquaredEuclideanDistances(x, n, d, squaredEuclidianDistances);

    // compute pairwise affinities
    getPairwiseAffinity(squaredEuclidianDistances, n, perp, affinity);

    free(squaredEuclidianDistances);

    // compute symmetric affinities
    float normalization_factor = 1/(2*n);
    for(int i = 0; i < n*n; i += n) {
        for(int j = i+1; j < n; j++) {
            float p_ij = affinity[n*i + j];
            float p_ji = affinity[n*j + i];

            float symmetric_affinity = (p_ij + p_ji) * normalization_factor;
            affinity[n*i + j] = symmetric_affinity;
            affinity[n*j + i] = symmetric_affinity;
        }
    }
}

// Compute squared Euclidean distance matrix
void getSquaredEuclideanDistances(float* X, int N, int D, float* DD) {
    const float* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const float* XmD = XnD + D;
        float* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}