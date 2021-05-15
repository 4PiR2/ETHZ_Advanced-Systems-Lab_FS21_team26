#include <stdio.h>
#include "tsne.h"

void gradientDescent(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            for(int k=0; k<n; k++){
                grad_cy[i*d+j] += (p[i*n+k]-t[i*n+k]/sum_t) * (y[i*d+j]-y[k*d+j]) * t[i*n+k];
            }
            grad_cy[i*d+j] *= 4;
            dy[i*d+j] = -eta * grad_cy[i*d+j] + alpha * dy[i*d+j];
            grad_cy[i*d+j] = 0;
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void compute_t(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            float diff, dist = 0;
            for(int k=0; k<d; k++){
                diff = y[i*d+k] - y[j*d+k];
                dist += diff * diff;
            }
            t[i*n+j] = 1/(1+dist);
        }
    }
}

float compute_sum_t(float* t, int n) {
    float sum_t = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i!=j) sum_t += t[i*n+j];
        }
    }
    return sum_t;
}

void getLowDimResult(float* y, float* dy, float* grad_cy, float* p, float* t, int n_samples, int d_out, float alpha, float eta, int n_iter) {
    for (int i = 0; i < n_iter; i++) {
		compute_t(y, t, n_samples, d_out);
		float sum_t = compute_sum_t(t, n_samples);
		gradientDescent(y, dy, grad_cy, p, t, sum_t, n_samples, d_out, alpha, eta);
	}
}
