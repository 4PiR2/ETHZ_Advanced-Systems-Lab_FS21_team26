#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <cmath>

// define the maximum number of iterations to fit the perplexity
int MAX_ITERATIONS = 200;

// define the error tolerance for the perplexity
float ERROR_TOLERANCE = 1e-5f;

void _getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *DD) {
	for (int i = 0; i < n_samples; i++) {
		DD[i * n_samples + i] = 0.f;
		for (int j = i + 1; j < n_samples; j++) {
			float dist = x[i * d_in] - x[j * d_in];
			dist *= dist;
			for (int k = 1; k < d_in; k++) {
				float diff = x[i * d_in + k] - x[j * d_in + k];
				dist += diff * diff;
			}
			DD[j * n_samples + i] = DD[i * n_samples + j] = dist;
		}
	}
}

void _getPairwiseAffinity(float *DD, int n_samples, float perplexity, float *p) {
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;
	// compute affinities row by row
	for (int i = 0; i < n_samples; i++) {
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = DD[i * n_samples + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		// initialize beta values, beta := -.5f / (sigma * sigma)
		float beta = -1.f / maxv;
		float beta_max, beta_min;
		bool flag0 = true, flag1 = true;
		float sum;
		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k) {
			// compute the conditional Gaussian densities for point i
			sum = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				if (i != j) {
					float bd = beta * DD[i * n_samples + j];
					float gaussian_density = expf(bd);
					shannon_entropy -= bd * gaussian_density;
					sum += gaussian_density;
					p[i * n_samples + j] = gaussian_density;
				}
			}
			shannon_entropy = shannon_entropy / sum + logf(sum);
			if (shannon_entropy > lb && shannon_entropy < rb) {
				break;
			} else {
				if (shannon_entropy > log_perp) {
					beta_min = beta;
					flag1 = false;
					if (flag0)
						beta *= 2.f;
					else
						beta = (beta + beta_max) * .5f;
				} else {
					beta_max = beta;
					flag0 = false;
					if (flag1)
						beta *= .5f;
					else
						beta = (beta + beta_min) * .5f;
				}
			}
		}
		float sum_inv = 1.f / sum;
		// normalize the row
		for (int j = 0; j < n_samples; j++) {
			if (i != j) {
				p[i * n_samples + j] *= sum_inv;
			}
		}
		p[i * n_samples + i] = 0.f;
	}
}

void _symmetrizeAffinities(float *p, int n_samples) {
	auto p_sum_inv = .5f / float(n_samples);
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float p_ij = p[i * n_samples + j], p_ji = p[j * n_samples + i];
			p[j * n_samples + i] = p[i * n_samples + j] = (p_ij + p_ji) * p_sum_inv;
		}
	}
}

#endif //PREPROCESSING_H
