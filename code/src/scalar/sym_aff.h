#ifndef SYM_AFF_H
#define SYM_AFF_H

#include "../baseline/pre.h"
#include <immintrin.h>

inline void swapptr(float** a, float** b) {
	float* tmp = *a;
	*a = *b;
	*b = tmp;
}

__m256 avx2_half = _mm256_set1_ps(.5f);

inline void avx2_scalar_mul(float* x, float a, float* ret, int size) {
	// for (int i = 0; i < size; i++) ret[i] = x[i] * a;

	__m256 scalar = _mm256_set1_ps(a);
	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 mul = _mm256_mul_ps(a, scalar);
		_mm256_store_ps(ret + i, mul);
	}
	for (; i < size; i++)
		ret[i] = x[i] * a;
}

inline void avx2_square(float* x, float* ret, int size) {
	// for (int i = 0; i < size; i++) ret[i] = x[i] * x[i];

	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 mul = _mm256_mul_ps(a, a);
		_mm256_store_ps(ret + i, mul);
	}
	for (; i < size; i++)
		ret[i] = x[i] * x[i];
}

inline void avx2_square_root(float* x, float* ret, int size) {
	// for (int i = 0; i < size; i++) ret[i] = sqrtf(x[i]);
	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 sqrt = _mm256_sqrt_ps(a);
		_mm256_store_ps(ret + i, sqrt);
	}
	for (; i < size; i++)
		ret[i] = sqrtf(x[i]);
}

inline void avx2_geometric_mean(float* x, float* y, float* ret, int size) {
	// for (int i = 0; i < size; i++) ret[i] = sqrtf(x[i] * y[i]);
	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 b = _mm256_load_ps(y + i);
		__m256 mul = _mm256_mul_ps(a, b);
		__m256 sqrt = _mm256_sqrt_ps(mul);
		_mm256_store_ps(ret + i, sqrt);
	}
	for (; i < size; i++)
		ret[i] = sqrtf(x[i] * y[i]);
}

inline void avx2_arithmetic_mean(float* x, float* y, float* ret, int size) {
	// for (int i = 0; i < size; i++) ret[i] = (x[i] + y[i]) * .5f;
	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 b = _mm256_load_ps(y + i);
		__m256 sum = _mm256_add_ps(a, b);
		__m256 mean = _mm256_mul_ps(sum, avx2_half);
		_mm256_store_ps(ret + i, mean);
	}
	for (; i < size; i++)
		ret[i] = (x[i] + y[i]) * .5f;
}

inline float avx2_sum_m256(__m256 x) {
	// https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

inline void avx2_dp_sum(float* x, float* y, float* ret_dp, float* ret_sum, int size) {
	// dot product x, y => ret_dp
	// sum x => ret_sum
	// for (int i = 0; i < size; i++) {
	// 	*ret_dp += x[i] * y[i];
	// 	*ret_sum += x[i];
	// }
	int i;
	__m256 sum = _mm256_setzero_ps();
	__m256 dp = _mm256_setzero_ps();

	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 b = _mm256_load_ps(y + i);
		__m256 mul = _mm256_mul_ps(a, b);
		dp = _mm256_add_ps(dp, mul);
		sum = _mm256_add_ps(sum, a);
	}
	*ret_dp = avx2_sum_m256(dp);
	*ret_sum = avx2_sum_m256(sum);

	for (; i < size; i++) {
		*ret_dp += x[i] * y[i];
		*ret_sum += x[i];
	}
}

// deleting the exp functions
#define SYM_AFF_PA_SCALAR_INIT // 1.83, 2.08
// deleting one branch
// #define SYM_AFF_PA_SCALAR_UP1 // 2.33 2.62
// deleting beta values
// #define SYM_AFF_PA_SCALAR_UP2 // 2.11
// try avx2 functions
// #define SYM_AFF_PA_AVX2 // 7.0(no-vec flag) 1.8(normal)
// #define SYM_AFF_PA_SCALAR_CURRENT 

#ifdef SYM_AFF_SQEU_DIST

void getSquaredEuclideanDistances(float *x, int n_samples, int d_in, float *d) {
	float* norms = (float*)malloc(n_samples * sizeof(float));
	for (int i = 0; i < n_samples; i++) {
		float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
		tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
		int k = 0;
		for (; k < d_in; k += 8) {
			int id = i * d_in + k;
			tmp0 += x[id + 0] * x[id + 0];
			tmp1 += x[id + 1] * x[id + 1];
			tmp2 += x[id + 2] * x[id + 2];
			tmp3 += x[id + 3] * x[id + 3];
			tmp4 += x[id + 4] * x[id + 4];
			tmp5 += x[id + 5] * x[id + 5];
			tmp6 += x[id + 6] * x[id + 6];
			tmp7 += x[id + 7] * x[id + 7];
		}

		tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
		for (; k < d_in; k++) {
			tmp += x[i * d_in + k] * x[i * d_in + k];
		}
		norms[i] = tmp;
	}

	for (int i = 0; i < n_samples; i++) {
		for (int j = i + 1; j < n_samples; j++) {
			float tmp, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
			tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.f;
			int k = 0;
			for (; k < d_in; k += 8) {
				int id = i * d_in + k;
				int jd = j * d_in + k;
				tmp0 += x[id + 0] * x[jd + 0];
				tmp1 += x[id + 1] * x[jd + 1];
				tmp2 += x[id + 2] * x[jd + 2];
				tmp3 += x[id + 3] * x[jd + 3];
				tmp4 += x[id + 4] * x[jd + 4];
				tmp5 += x[id + 5] * x[jd + 5];
				tmp6 += x[id + 6] * x[jd + 6];
				tmp7 += x[id + 7] * x[jd + 7];
			}

			tmp = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7;
			for (; k < d_in; k++) {
				tmp += x[i * d_in + k] * x[j * d_in + k];
			}

			d[i * n_samples + j] = norms[i] - 2*tmp + norms[j];
		}
	}

	free(norms);
}

#else

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

#endif //SYM_AFF_SQEU_DIST

#ifdef SYM_AFF_PA_SCALAR_INIT

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
	}
	
	free(bd_exps), free(bd_max_exps), free(bd_min_exps);
}

#endif // SYM_AFF_SCALAR_INIT

#ifdef SYM_AFF_PA_SCALAR_UP1

void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	printf("CURRENT\n");
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
			bd_exps[i] = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				float bd = beta * d[i * n_samples + j];
				float gaussian_density = bd_exps[j];
				shannon_entropy -= bd * gaussian_density;
				sum += gaussian_density;
				p[i * n_samples + j] = gaussian_density;
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

	free(bd_exps), free(bd_max_exps), free(bd_min_exps);
}

#endif // SYM_AFF_SCALAR_UP1

#ifdef SYM_AFF_PA_SCALAR_UP2
// will modify euclidean distance file
void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	printf("CURRENT\n");
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;
	
	float* bd_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_max_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_min_exps = (float*)malloc(n_samples * sizeof(float));
	float* bds = (float*)malloc(n_samples * sizeof(float));
	float* bd_maxs = (float*)malloc(n_samples * sizeof(float));
	float* bd_mins = (float*)malloc(n_samples * sizeof(float));

	for (int i = 0; i < n_samples; i++) {
		int row = i * n_samples;
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = d[row + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		float beta = -1.f / maxv, sum;
		for (int j = 0; j < n_samples; j++) {
			bds[j] = beta * d[row + j];
			bd_exps[j] = expf(bds[j]);
		}
		bool flag0 = true, flag1 = true;

		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k) {
			// compute the conditional Gaussian densities for point i
			sum = 0.f;
			bds[i] = bd_exps[i] = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				float bd = bds[j];
				float gaussian_density = bd_exps[j];
				shannon_entropy -= bd * gaussian_density;
				sum += gaussian_density;
			}
			shannon_entropy = shannon_entropy / sum + logf(sum);
			if (shannon_entropy > lb && shannon_entropy < rb) {
				break;
			}
			if (shannon_entropy > log_perp) {
				swapptr(&bd_exps, &bd_min_exps);
				swapptr(&bds, &bd_mins);
			}
			else {
				swapptr(&bd_exps, &bd_max_exps);
				swapptr(&bds, &bd_maxs);
			}
			if (flag0 || flag1) {
				if (shannon_entropy > log_perp) {
					flag1 = false;
					if (flag0) {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = bd_min_exps[j] * bd_min_exps[j];
							bds[j] = 2.f * bd_mins[j];		
						}
					}
					else {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_min_exps[j] * bd_max_exps[j]);
							bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
						}
					}
				} else {
					flag0 = false;
					if (flag1) {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_max_exps[j]);
							bds[j] = .5f * bd_maxs[j];
						}
					}
					else {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_max_exps[j] * bd_min_exps[j]);
							bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
						}
					}
				}
			}
			else {
				for (int j = 0; j < n_samples; j++) {
					bd_exps[j] = sqrtf(bd_max_exps[j] * bd_min_exps[j]);
					bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
				}
			}
			
		}
	}

	free(bd_exps), free(bd_max_exps), free(bd_min_exps), free(bds), free(bd_maxs), free(bd_mins);
}
#endif // SYM_AFF_SCALAR_UP2

// Replace the 'expf' function call by an approximation
// Improve stopping criteria
#ifdef SYM_AFF_PA_SCALAR_UP3

// Approximation of the exponential function by linear regression (polynomial of order three)
inline __m512 exp_app_ps(__m512 x) {
	// -1 <= x <= 0
	__m512 y,
	a0 = _mm512_set1_ps(.9996136409397813f),
	a1 = _mm512_set1_ps(.9920487460431511f),
	a2 = _mm512_set1_ps(.4624692123106021f),
	a3 = _mm512_set1_ps(.10250045262707179f);
	y = _mm512_fmadd_ps(a3, x, a2);
	y = _mm512_fmadd_ps(y, x, a1);
	y = _mm512_fmadd_ps(y, x, a0);
	return y;
}

void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	printf("CURRENT\n");
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;
	
	float* bd_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_max_exps = (float*)malloc(n_samples * sizeof(float));
	float* bd_min_exps = (float*)malloc(n_samples * sizeof(float));
	float* bds = (float*)malloc(n_samples * sizeof(float));
	float* bd_maxs = (float*)malloc(n_samples * sizeof(float));
	float* bd_mins = (float*)malloc(n_samples * sizeof(float));

	for (int i = 0; i < n_samples; i++) {
		int row = i * n_samples;
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = d[row + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		float beta = -1.f / maxv, sum;
		for (int j = 0; j < n_samples; j++) {
			bds[j] = beta * d[row + j];
			bd_exps[j] = exp_app(bds[j]);
		}
		bool flag0 = true, flag1 = true;

		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k) {
			// compute the conditional Gaussian densities for point i
			sum = 0.f;
			bds[i] = bd_exps[i] = 0.f;
			float shannon_entropy = 0.f;
			for (int j = 0; j < n_samples; j++) {
				float bd = bds[j];
				float gaussian_density = bd_exps[j];
				shannon_entropy -= bd * gaussian_density;
				sum += gaussian_density;
			}
			shannon_entropy = shannon_entropy / sum + logf(sum);
			if (shannon_entropy > lb && shannon_entropy < rb) {
				break;
			}
			if (shannon_entropy > log_perp) {
				swapptr(&bd_exps, &bd_min_exps);
				swapptr(&bds, &bd_mins);
			}
			else {
				swapptr(&bd_exps, &bd_max_exps);
				swapptr(&bds, &bd_maxs);
			}
			if (flag0 || flag1) {
				if (shannon_entropy > log_perp) {
					flag1 = false;
					if (flag0) {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = bd_min_exps[j] * bd_min_exps[j];
							bds[j] = 2.f * bd_mins[j];		
						}
					}
					else {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_min_exps[j] * bd_max_exps[j]);
							bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
						}
					}
				} else {
					flag0 = false;
					if (flag1) {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_max_exps[j]);
							bds[j] = .5f * bd_maxs[j];
						}
					}
					else {
						for (int j = 0; j < n_samples; j++) {
							bd_exps[j] = sqrtf(bd_max_exps[j] * bd_min_exps[j]);
							bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
						}
					}
				}
			}
			else {
				for (int j = 0; j < n_samples; j++) {
					bd_exps[j] = sqrtf(bd_max_exps[j] * bd_min_exps[j]);
					bds[j] = (bd_mins[j] + bd_maxs[j]) * .5f;
				}
			}
			
		}
	}

	free(bd_exps), free(bd_max_exps), free(bd_min_exps), free(bds), free(bd_maxs), free(bd_mins);
}
#endif // SYM_AFF_SCALAR_UP3

#ifdef SYM_AFF_PA_AVX2
// will modify euclidean distance file
void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	printf("AVX2\n");
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;

	float* bd_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_max_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_min_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bds = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_maxs = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_mins = (float*)aligned_alloc(256, n_samples * sizeof(float));	

	for (int i = 0; i < n_samples; i++) {
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = d[i * n_samples + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		float beta = -1.f / maxv, sum;
		for (int j = 0; j < n_samples; j++) {
			bds[j] = beta * d[i * n_samples + j];
			bd_exps[j] = expf(bds[j]);
			bds[j] = -bds[j];
		}
		bool flag0 = true, flag1 = true;

		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k) {
			// compute the conditional Gaussian densities for point i
			sum = 0.f;
			bds[i] = bd_exps[i] = 0.f;
			float shannon_entropy = 0.f;

			avx2_dp_sum(bd_exps, bds, &shannon_entropy, &sum, n_samples);

			shannon_entropy = shannon_entropy / sum + logf(sum);
			if (shannon_entropy > lb && shannon_entropy < rb) {
				break;
			}
			if (shannon_entropy > log_perp) {
				swapptr(&bd_exps, &bd_min_exps);
				swapptr(&bds, &bd_mins);
			}
			else {
				swapptr(&bd_exps, &bd_max_exps);
				swapptr(&bds, &bd_maxs);
			}
			if (flag0 || flag1) {
				if (shannon_entropy > log_perp) {
					flag1 = false;
					if (flag0) {
						avx2_square(bd_min_exps, bd_exps, n_samples);
						avx2_scalar_mul(bd_mins, 2.f, bds, n_samples);
					}
					else {
						avx2_geometric_mean(bd_min_exps, bd_max_exps, bd_exps, n_samples);
						avx2_arithmetic_mean(bd_mins, bd_maxs, bds, n_samples);
					}
				} else {
					flag0 = false;
					if (flag1) {
						avx2_square_root(bd_max_exps, bd_exps, n_samples);
						avx2_scalar_mul(bd_maxs, .5f, bds, n_samples);
					}
					else {
						avx2_geometric_mean(bd_min_exps, bd_max_exps, bd_exps, n_samples);
						avx2_arithmetic_mean(bd_mins, bd_maxs, bds, n_samples);
					}
				}
			}
			else {
				avx2_geometric_mean(bd_min_exps, bd_max_exps, bd_exps, n_samples);
				avx2_arithmetic_mean(bd_mins, bd_maxs, bds, n_samples);
			}
			
		}
	}
	
	free(bd_exps), free(bd_max_exps), free(bd_min_exps), free(bds), free(bd_maxs), free(bd_mins);
}
#endif // SYM_AFF_SCALAR_AVX2

#endif //SYM_AFF_H
