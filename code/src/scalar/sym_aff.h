#ifndef SYM_AFF_H
#define SYM_AFF_H

#include "../baseline/pre.h"

void getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *d) {
	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float tmp;
			float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
			int k = 0;
			for (; k < d_in; k += 8) {
				float sq0, sq1, sq2, sq3, sq4, sq5, sq6, sq7;
				int id = i * d_in + k, jd = j * d_in + k;
				sq0 = x[id + 0] - x[jd + 0];
				sq1 = x[id + 1] - x[jd + 1];
				sq2 = x[id + 2] - x[jd + 2];
				sq3 = x[id + 3] - x[jd + 3];
				sq4 = x[id + 4] - x[jd + 4];
				sq5 = x[id + 5] - x[jd + 5];
				sq6 = x[id + 6] - x[jd + 6];
				sq7 = x[id + 7] - x[jd + 7];

				tmp0 += sq0 * sq0;
				tmp1 += sq1 * sq1;
				tmp2 += sq2 * sq2;
				tmp3 += sq3 * sq3;
				tmp4 += sq4 * sq4;
				tmp5 += sq5 * sq5;
				tmp6 += sq6 * sq6;
				tmp7 += sq7 * sq7;
			}

			tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
			for (; k < d_in; k++) {
				float sq = x[i * d_in + k] - x[j * d_in + k];
				tmp += sq * sq;
			}
			d[i * n_samples + j] = d[j * n_samples + i] = tmp;
		}
		d[i * n_samples + i] = 0.f;
	}
}

void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;
	
	float* bd_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_max_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_min_exps = (float*)malloc(n_samples * sizeof(float));

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
		float beta = -1.f / maxv, beta_max, beta_min, sum;
		// change all betas to exps
		for (int j = 0; j < n_samples; j++) {
			float dist = d[i * n_samples + j];
			bd_exps[j] = expf(beta * dist);
		}
		bool flag0 = true, flag1 = true;
		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k) {
			// compute the conditional Gaussian densities for point i
			sum = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				if (i != j) {
					float bd = beta * d[i * n_samples + j];
					float gaussian_density = bd_exps[j];
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
					float* tmp = bd_exps;
					bd_exps = bd_min_exps;
					bd_min_exps = tmp;

					flag1 = false;
					if (flag0) {
						beta *= 2.f;
						for (int j = 0; j < n_samples; j++)
							bd_exps[j] = tmp[j] * tmp[j];
					}
					else {
						beta = (beta + beta_max) * .5f;
						for (int j = 0; j < n_samples; j++)
							bd_exps[j] = sqrtf(tmp[j] * bd_max_exps[j]);
					}
				} else {
					beta_max = beta;
					float* tmp = bd_exps;
					bd_exps = bd_max_exps;
					bd_max_exps = tmp;

					flag0 = false;
					if (flag1) {
						beta *= .5f;
						for (int j = 0; j < n_samples; j++)
							bd_exps[j] = sqrtf(tmp[j]);
					}
					else {
						beta = (beta + beta_min) * .5f;
						for (int j = 0; j < n_samples; j++)
							bd_exps[j] = sqrtf(tmp[j] * bd_min_exps[j]);
					}
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

#endif //SYM_AFF_H
