#include <stdio.h>
#include <math.h>
#include <random>
#include <tuple>
#include <limits>
#include "tsne.h"
#include "baselines.h"
#include "benchmark.h"

// define the maximum number of iterations to fit the perplexity
int MAX_ITERATIONS = 200;

// define the error tolerance for the perplexity
float ERROR_TOLERANCE = 1e-5;

float MIN_FLOAT = std::numeric_limits<float>::min();
float MAX_FLOAT = std::numeric_limits<float>::max();

// function declarations
void getPairwiseAffinity(float* squaredEuclidianDistances, int n_samples, float perplexity, float* affinity);
void normalizeData(float* X, int n_samples, int d_in);
void symmetrizeAffinities(float* P, int n_samples);
std::tuple<float, float, float> updateBetaValues(float entropy_error, float beta_min, float beta_max, float beta);


void getSymmetricAffinity(float* X, int n_samples, int d_in, float perp, float* P, float* ED) {
    //normalizeData(x, n, d);
    thandle t1 = create_timer("ED"), t2 = create_timer("_ED");
    start(t1);
	getSquaredEuclideanDistances(X, n_samples, d_in, ED);
    stop(t1);
    // baseline
    float* _ED = mat_alloc<float>(n_samples, n_samples);
    start(t2);
    _getSquaredEuclideanDistances(X, n_samples, d_in, _ED);
    stop(t2);
    baselineCompare(ED, _ED, n_samples * n_samples, "ED baseline compare");

    // compute pairwise affinities
    t1 = create_timer("PA"), t2 = create_timer("_PA");
    start(t1);
    getPairwiseAffinity(ED, n_samples, perp, P);
    stop(t1);
    // baseline
    float* _P = mat_alloc<float>(n_samples, n_samples);
    start(t2);
    _getPairwiseAffinity(ED, n_samples, perp, _P);
    stop(t2);
    baselineCompare(P, _P, n_samples * n_samples, "PA baseline compare");
    
    t1 = create_timer("SA"), t2 = create_timer("_SA");
    start(t1);
    symmetrizeAffinities(P, n_samples);
    stop(t1);
    // baseline
    start(t2);
    symmetrizeAffinities(_P, n_samples);
    stop(t2);
    baselineCompare(P, _P, n_samples * n_samples, "SA baseline compare");
    
}


// #define BASELINE
#define SCALAROP

#ifdef BASELINE
void getSquaredEuclideanDistances(float* X, int n_samples, int dim, float* DD) {
    const float* XnD = X;
    for(int n = 0; n < n_samples; ++n, XnD += dim) {
        const float* XmD = XnD + dim;
        float* curr_elem = &DD[n*n_samples + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + n_samples;
        for(int m = n + 1; m < n_samples; ++m, XmD+=dim, curr_elem_sym+=n_samples) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < dim; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}
#endif

#ifdef SCALAROP
void getSquaredEuclideanDistances(float* X, int n_samples, int dim, float* DD) {
    // const float* XnD = X;
    // for(int n = 0; n < n_samples; ++n, XnD += dim) {
    //     const float* XmD = XnD + dim;
    //     float* curr_elem = &DD[n*n_samples + n];
    //     *curr_elem = 0.0;
    //     float* curr_elem_sym = curr_elem + n_samples;
    //     for(int m = n + 1; m < n_samples; ++m, XmD+=dim, curr_elem_sym+=n_samples) {
    //         *(++curr_elem) = 0.0;
    //         for(int d = 0; d < dim; ++d) {
    //             *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
    //         }
    //         *curr_elem_sym = *curr_elem;
    //     }
    // }

    // for (int i = 0; i < n_samples; i++) {
    //     for (int j = i + 1; j < n_samples; j++) {
    //         float tmp = 0.0;
    //         for (int k = 0; k < dim; k++) {
    //             float sq = X[i * dim + k] - X[j * dim + k];
    //             tmp += sq * sq;
    //         }
    //         DD[i * n_samples + j] = DD[j * n_samples + i] = tmp;
    //     }
    //     DD[i * n_samples + i] = 0.0;
    // }

    for (int i = 0; i < n_samples; i++) {
        for (int j = i + 1; j < n_samples; j++) {
            float tmp;
            float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
            tmp = tmp0 = tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = tmp7 = 0.0;
            int k = 0;
            for (; k < dim; k += 8) {
                float sq0, sq1, sq2, sq3, sq4, sq5, sq6, sq7;
                int id = i * dim + k, jd = j * dim + k;
                sq0 = X[id + 0] - X[jd + 0];
                sq1 = X[id + 1] - X[jd + 1];
                sq2 = X[id + 2] - X[jd + 2];
                sq3 = X[id + 3] - X[jd + 3];
                sq4 = X[id + 4] - X[jd + 4];
                sq5 = X[id + 5] - X[jd + 5];
                sq6 = X[id + 6] - X[jd + 6];
                sq7 = X[id + 7] - X[jd + 7];

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
            for (; k < dim; k++) {
                float sq = X[i * dim + k] - X[j * dim + k];
                tmp += sq * sq;
            }
            DD[i * n_samples + j] = DD[j * n_samples + i] = tmp;
        }
        DD[i * n_samples + i] = 0.0;
    }
}
#endif


void getPairwiseAffinity(float* squaredEuclidianDistances, int n_samples, float perplexity, float* P) {
    float log_perp = logf(perplexity);

    // compute affinities row by row
    for (int i = 0; i < n_samples; i++) {
        // initialize beta values, beta := -.5f / (sigma * sigma)
        float beta = 1.0;
        float beta_max = MAX_FLOAT;
        float beta_min = MIN_FLOAT;

        float sum = 0.0;
        // perform binary search to find the optimal beta values for each data point
        for (int k = 0; k < MAX_ITERATIONS; ++k) {

            // compute the conditional Gaussian densities for point i
            sum = 0.0;
			for(int j = 0; j < n_samples; j++) {
                float gaussian_density = expf(-beta * squaredEuclidianDistances[n_samples*i + j]);
			    P[n_samples*i + j] = gaussian_density;
                // if (i != j)
                    sum += gaussian_density;
			}

			float shannon_entropy = 0.0;
			for (int j = 0; j < n_samples; j++) shannon_entropy += beta * (squaredEuclidianDistances[n_samples*i + j] * P[n_samples*i + j]);
			shannon_entropy = (shannon_entropy / sum) + logf(sum);

			float entropy_error = shannon_entropy - log_perp;
			if (fabs(entropy_error) < ERROR_TOLERANCE) {
                break;
			} else {
                auto betaValues = updateBetaValues(entropy_error, beta_min, beta_max, beta);
                beta_min = std::get<0>(betaValues);
                beta_max = std::get<1>(betaValues);
                beta = std::get<2>(betaValues);
			}
        }

        sum--;
        // normalize the row
        for(int j = 0; j < n_samples; j++) {
            P[n_samples*i + j] /= sum;
            P[n_samples*i + j] *= 4.f;
        }
        P[n_samples*i + i] =0.f;
    }
}

std::tuple<float, float, float> updateBetaValues(float entropy_error, float beta_min, float beta_max, float beta) {
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

    return std::make_tuple(beta_min, beta_max, beta);
}

void symmetrizeAffinities(float* P, int n_samples) {
    for(int i = 0; i < n_samples; i++) {
        int mN = (i + 1) * n_samples;
        for(int j = i + 1; j < n_samples; j++) {
            P[i*n_samples + j] += P[mN + i];
            P[mN + i]  = P[i*n_samples + j];
            mN += n_samples;
        }
    }

    float sum_P = 0.f;
    for(int i = 0; i < n_samples * n_samples; i++) {
        sum_P += P[i];
    }

    for(int i = 0; i < n_samples * n_samples; i++) {
        P[i] /= sum_P;
    }
}

/**
 * @brief  Compute z-scores of data points
 *
 * @param  X: input data
 * @param  n_samples: number of data point
 * @param  d_in: data point dimension
 * @retval None
 */
void normalizeData(float* X, int n_samples, int d_in) {
    float* mean = (float*) calloc(d_in, sizeof(float));
    float* var = (float*) calloc(d_in, sizeof(float));

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < d_in; j++) {
            mean[j] +=  X[i*n_samples + j];
        }
    }

    for (int k = 0; k < d_in; k++) {
        mean[k] /= n_samples;
    }

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < d_in; j++) {
            float diff =  X[i*n_samples + j] - mean[j];
            X[i*n_samples + j] = diff * diff;
            var[j] += diff;
        }
    }

    for (int k = 0; k < d_in; k++) {
        var[k] = sqrt(var[k] / n_samples);
    }
    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < d_in; j++) {
            X[i*n_samples + j] = X[i*n_samples + j] / var[j];
        }
    }
}
