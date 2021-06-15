#ifndef SYM_AFF_H
#define SYM_AFF_H

#include <immintrin.h>

#define MAX_ITERATIONS 200
#define ERROR_TOLERANCE 1e-5f

inline void swapptr(float** a, float** b) {
	float* tmp = *a;
	*a = *b;
	*b = tmp;
}

__m256 avx2_half = _mm256_set1_ps(.5f);

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

// TODO Loop UnRolling...
inline void avx2_gmean_sum_dp(float* x, float* y, float* z, float* ret, float* ret_dp, float* ret_sum, int size) {
	__m256 sum = _mm256_setzero_ps();
	__m256 dp = _mm256_setzero_ps();

	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 b = _mm256_load_ps(y + i);
		__m256 c = _mm256_load_ps(z + i);

		__m256 mul = _mm256_mul_ps(a, b);
		__m256 sqrt = _mm256_sqrt_ps(mul);

		_mm256_store_ps(ret + i, sqrt);
		dp = _mm256_fmadd_ps(sqrt, c, dp);
		sum = _mm256_add_ps(sum, sqrt);
	}
	*ret_dp = avx2_sum_m256(dp);
	*ret_sum = avx2_sum_m256(sum);

	for (; i < size; i++) {
		ret[i] = sqrtf(x[i] * y[i]);
		*ret_dp += ret[i] * z[i];
		*ret_sum += ret[i];
	}
}

inline void avx2_square_sum_dp(float* x, float* y, float* ret, float* ret_dp, float* ret_sum, int size) {
	__m256 sum = _mm256_setzero_ps();
	__m256 dp = _mm256_setzero_ps();

	int i;
	for (i = 0; i + 7 < size; i += 8) {
		__m256 a = _mm256_load_ps(x + i);
		__m256 b = _mm256_load_ps(y + i);
		__m256 mul = _mm256_mul_ps(a, a);
		_mm256_store_ps(ret + i, mul);
		dp = _mm256_fmadd_ps(mul, b, dp);
		sum = _mm256_add_ps(sum, mul);
	}
	*ret_dp = avx2_sum_m256(dp);
	*ret_sum = avx2_sum_m256(sum);

	for (; i < size; i++) {
		ret[i] = x[i] * x[i];
		*ret_dp += ret[i] * y[i];
		*ret_sum += ret[i];
	}
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
		dp = _mm256_fmadd_ps(a, b, dp);
		sum = _mm256_add_ps(sum, a);
	}
	*ret_dp = avx2_sum_m256(dp);
	*ret_sum = avx2_sum_m256(sum);

	for (; i < size; i++) {
		*ret_dp += x[i] * y[i];
		*ret_sum += x[i];
	}
}

// will modify euclidean distance file
void getPairwiseAffinity(float *d, int n_samples, float perplexity, float *p) {
	float log_perp = logf(perplexity), lb = log_perp - ERROR_TOLERANCE, rb = log_perp + ERROR_TOLERANCE;

	float* bd_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_max_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));
	float* bd_min_exps = (float*)aligned_alloc(256, n_samples * sizeof(float));

	int cnt_loops = 0;

	for (int i = 0; i < n_samples; i++) {
		float maxv = std::numeric_limits<float>::min();
		for (int j = 0; j < n_samples; j++) {
			float dist = d[i * n_samples + j];
			if (dist > maxv) {
				maxv = dist;
			}
		}
		float beta = -1.f / maxv, beta_min, beta_max, sum, shannon_entropy;
		for (int j = 0; j < n_samples; j++) {
			bd_exps[j] = expf(beta * d[i * n_samples + j]);
		}
		bool flag0 = true, flag1 = true;
		// perform binary search to find the optimal beta values for each data point
		for (int k = 0; k < MAX_ITERATIONS; ++k, ++cnt_loops) {
			// compute the conditional Gaussian densities for point i
			// sum = 0.f;
			bd_exps[i] = 0.f;

			if (k == 0) {
				avx2_dp_sum(bd_exps, d + i * n_samples, &shannon_entropy, &sum, n_samples);
			}

			shannon_entropy = -shannon_entropy * beta / sum + logf(sum);
			if (shannon_entropy > lb && shannon_entropy < rb) {
				break;
			}

			if (shannon_entropy > log_perp) {
				swapptr(&bd_exps, &bd_min_exps);
				beta_min = beta;
				flag1 = false;
				if (flag0) {
					avx2_square_sum_dp(bd_min_exps, d + i * n_samples, bd_exps, &shannon_entropy, &sum, n_samples);
					beta *= 2.f;
				}
				else {
					avx2_gmean_sum_dp(bd_min_exps, bd_max_exps, d + i * n_samples, bd_exps, &shannon_entropy, &sum, n_samples);
					beta = (beta + beta_max) * .5f;
				}
			}
			else {
				swapptr(&bd_exps, &bd_max_exps);
				beta_max = beta;
				flag0 = false;
				if (flag1) {
					avx2_square_root(bd_max_exps, bd_exps, n_samples);
					beta *= .5f;
					avx2_dp_sum(bd_exps, d + i * n_samples, &shannon_entropy, &sum, n_samples);
				}
				else {
					avx2_gmean_sum_dp(bd_min_exps, bd_max_exps, d + i * n_samples, bd_exps, &shannon_entropy, &sum, n_samples);
					beta = (beta + beta_min) * .5f;
				}
			}

		}

		float sum_inv = 1.f / sum;

		for (int j = 0; j < n_samples; j++) {
			if (i != j) {
				p[i * n_samples + j] = bd_exps[j] * sum_inv;
			}
		}
		p[i * n_samples + i] = 0.f;
	}
	#ifdef SYM_AFF_COUNT_LOOPS
	printf("LOOPS:%d\n", cnt_loops);
	#endif
}

#endif //SYM_AFF_H
