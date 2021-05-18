#include <assert.h>
#include <tuple>

#include "baselines.h"
#include "math.h"

#define DEBUG

#define eps_baselines 1e-7
// C is the assertion message
#define assertEq(A,B,C) assert(fabs(A - B) <= eps_baselines && C) 

void baselineCompare(const float* X, const float* Y, const int size, const char* msg) {
#ifdef DEBUG
    for (int i = 0; i < size; i++)
        assertEq(X[i], Y[i], msg);
#endif
}

void _getSquaredEuclideanDistances(const float* X, int n_samples, int dim, float* DD) {
#ifdef DEBUG
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
#endif
}

// define the maximum number of iterations to fit the perplexity
extern int MAX_ITERATIONS;// = 200;

// define the error tolerance for the perplexity
extern float ERROR_TOLERANCE;// = 1e-5;

extern float MIN_FLOAT;// = std::numeric_limits<float>::min();
extern float MAX_FLOAT;//  = std::numeric_limits<float>::max();

static std::tuple<float, float, float> updateBetaValues(float entropy_error, float beta_min, float beta_max, float beta) {
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

void _getPairwiseAffinity(const float* squaredEuclidianDistances, int n_samples, float perplexity, float* P) {
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