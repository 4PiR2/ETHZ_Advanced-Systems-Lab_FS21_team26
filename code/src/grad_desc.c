#include <stdio.h>

void Gradient_descent(double* y, double* dy, double* grad_cy, double* p, double* t, double sum_t, int n, int d, int alpha, int eta) {
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            for(int k=0; i<n; i++){
                grad_cy[i*d+j] += (p[i*n+k]-t[i*n+k]/sum_t) * (y[i*d+j]-y[k*d+j]) * t[i*n+k];
            }
            grad_cy[i*d+j] *= 4;
            dy[i*d+j] = eta * grad_cy[i*d+j] + alpha * dy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void compute_t(double* y, double* t, int n, int d){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            double diff, dist = 0;
            for(int k=0; k<d; k++){
                diff = y[i*d+k] - y[j*d+k];
                dist += diff * diff;
            }
            t[i*n+j] = 1/(1+dist);
        }
    }
}

double compute_sum_t(double* t, int n){
    double sum_t = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i!=j) sum_t += t[i*n+j];
        }
    }
    return sum_t;
}
