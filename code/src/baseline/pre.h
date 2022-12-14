#ifndef PRE_H
#define PRE_H

#include <cmath>

void pre_pair_sq_dist(float *dummy0, float *d, float *x, float *dummy1, int n_samples, int d_in) {
	for (int i = 0; i < n_samples; i++) {
		d[i * n_samples + i] = 0.f;
		for (int j = i + 1; j < n_samples; j++) {
			float dist = x[i * d_in] - x[j * d_in];
			dist *= dist;
			for (int k = 1; k < d_in; k++) {
				float diff = x[i * d_in + k] - x[j * d_in + k];
				dist += diff * diff;
			}
			d[j * n_samples + i] = d[i * n_samples + j] = dist;
		}
	}
}

void pre_unfold_low_tri(float *dummy0, int dummy1) {}

int pre_perplex_bi_search(float *p, float *d, float perplexity, float epsilon, float *dummy0, int n_samples) {
	float log_perp = logf(perplexity), lb = log_perp - epsilon, rb = log_perp + epsilon;
	int iter, count = 0;
	// compute affinities row by row
	for (int i = 0; i < n_samples; i++) {
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = d[i * n_samples + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		// initialize beta values, beta := -.5f / (sigma * sigma)
		float beta = -1.f / maxv, beta_max, beta_min, beta_last = 0.f, sum;
		bool flag0 = true, flag1 = true;
		// perform binary search to find the optimal beta values for each data point
		iter = 0;
		while (beta != beta_last) {
			// compute the conditional Gaussian densities for point i
			++iter;
			beta_last = beta;
			sum = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				if (i != j) {
					float bd = beta * d[i * n_samples + j];
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
		count += iter;
	}
	return count;
}

void pre_sym_aff(float *p, int n_samples) {
	auto p_sum_inv = .5f / float(n_samples);
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float p_ij = p[i * n_samples + j], p_ji = p[j * n_samples + i];
			p[j * n_samples + i] = p[i * n_samples + j] = (p_ij + p_ji) * p_sum_inv;
		}
	}
}

#endif //PRE_H
