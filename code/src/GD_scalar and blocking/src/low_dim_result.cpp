#include <stdio.h>
#include "tsne.h"

void gradientDescent1(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=0; k<n; k++){
                grad_cy[i*d+j] += (p[i*n+k]-t[i*n+k]/sum_t) * (y[i*d+j]-y[k*d+j]) * t[i*n+k];
            }
            grad_cy[i*d+j] *= 4;
            dy[i*d+j] = -eta * grad_cy[i*d+j] + alpha * dy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent2(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=0; k<n; k++){
                grad_cy[i*d+j] += (p[i*n+k]-t[i*n+k]/sum_t) * (y[i*d+j]-y[k*d+j]) * t[i*n+k];
            }
            dy[i*d+j] = eta * grad_cy[i*d+j] + alpha * dy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent3(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    float q;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=0; k<n; k++){
                q = t[i*n+k]/sum_t;
                float pt = p[i*n+k]-q;
                float y_diff = y[i*d+j]-y[k*d+j];
                float grad_add = pt * y_diff;
                grad_add *= t[i*n+k];
                grad_cy[i*d+j] += grad_add;
            }
            grad_cy[i*d+j] *= eta;
            dy[i*d+j] *= alpha;
            dy[i*d+j] += grad_cy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent_scalar1(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    float q1, q2;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=0; k<n; k+=2){
                q1 = t[i*n+k]/sum_t;
                float pt = p[i*n+k]-q1;
                float y_diff = y[i*d+j]-y[k*d+j];
                float grad_add = pt * y_diff;
                grad_add *= t[i*n+k];
                q2 = t[i*n+k+1]/sum_t;
                float pt2 = p[i*n+k+1]-q2;
                float y_diff2 = y[i*d+j]-y[k*d+j+d];
                float grad_add2 = pt2 * y_diff2;
                grad_add2 *= t[i*n+k+1];
                grad_cy[i*d+j] += grad_add;
                grad_cy[i*d+j] += grad_add2;
            }
            grad_cy[i*d+j] *= eta;
            dy[i*d+j] *= alpha;
            dy[i*d+j] += grad_cy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent_scalar2(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    float q1, q2;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=i+1; k<n; k+=2){
                q1 = t[i*n+k]/sum_t;
                float pt = p[i*n+k]-q1;
                float y_diff = y[i*d+j]-y[k*d+j];
                float grad_add = pt * y_diff;
                grad_add *= t[i*n+k];
                q2 = t[i*n+k+1]/sum_t;
                float pt2 = p[i*n+k+1]-q2;
                float y_diff2 = y[i*d+j]-y[k*d+j+d];
                float grad_add2 = pt2 * y_diff2;
                grad_add2 *= t[i*n+k+1];
                grad_cy[i*d+j] += grad_add;
                grad_cy[i*d+j] += grad_add2;
            }
            grad_cy[i*d+j] *= eta;
            dy[i*d+j] *= alpha;
            dy[i*d+j] += grad_cy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent_scalar3(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    float q1, q2, q3, q4;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=i+1; k<n; k+=4){
                q1 = t[i*n+k]/sum_t;
                float pt = p[i*n+k]-q1;
                float y_diff = y[i*d+j]-y[k*d+j];
                float grad_add = pt * y_diff;
                grad_add *= t[i*n+k];
                q2 = t[i*n+k+1]/sum_t;
                float pt2 = p[i*n+k+1]-q2;
                float y_diff2 = y[i*d+j]-y[k*d+j+d];
                float grad_add2 = pt2 * y_diff2;
                grad_add2 *= t[i*n+k+1];
                q3 = t[i*n+k+2]/sum_t;
                float pt3 = p[i*n+k+2]-q3;
                float y_diff3 = y[i*d+j]-y[(k+2)*d+j];
                float grad_add3 = pt3 * y_diff3;
                grad_add3 *= t[i*n+k+2];
                q4 = t[i*n+k+3]/sum_t;
                float pt4 = p[i*n+k+3]-q4;
                float y_diff4 = y[i*d+j]-y[(k+3)*d+j+d];
                float grad_add4 = pt4 * y_diff4;
                grad_add4 *= t[i*n+k+3];
                grad_cy[i*d+j] += grad_add;
                grad_cy[i*d+j] += grad_add2;
                grad_cy[i*d+j] += grad_add3;
                grad_cy[i*d+j] += grad_add4;
            }
            grad_cy[i*d+j] *= eta;
            dy[i*d+j] *= alpha;
            dy[i*d+j] += grad_cy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent_scalar4(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta) {
    eta *= -4;
    float q1, q2, q3, q4, q5, q6;
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            grad_cy[i*d+j] = 0;
            for(int k=i+1; k<n; k+=6){
                q1 = t[i*n+k]/sum_t;
                float pt = p[i*n+k]-q1;
                float y_diff = y[i*d+j]-y[k*d+j];
                float grad_add = pt * y_diff;
                grad_add *= t[i*n+k];
                q2 = t[i*n+k+1]/sum_t;
                float pt2 = p[i*n+k+1]-q2;
                float y_diff2 = y[i*d+j]-y[k*d+j+d];
                float grad_add2 = pt2 * y_diff2;
                grad_add2 *= t[i*n+k+1];
                q3 = t[i*n+k+2]/sum_t;
                float pt3 = p[i*n+k+2]-q3;
                float y_diff3 = y[i*d+j]-y[(k+2)*d+j];
                float grad_add3 = pt3 * y_diff3;
                grad_add3 *= t[i*n+k+2];
                q4 = t[i*n+k+3]/sum_t;
                float pt4 = p[i*n+k+3]-q4;
                float y_diff4 = y[i*d+j]-y[(k+3)*d+j+d];
                float grad_add4 = pt4 * y_diff4;
                grad_add4 *= t[i*n+k+3];
                q5 = t[i*n+k+4]/sum_t;
                float pt5 = p[i*n+k+4]-q5;
                float y_diff5 = y[i*d+j]-y[(k+4)*d+j+d];
                float grad_add5 = pt5 * y_diff5;
                grad_add5 *= t[i*n+k+4];
                q6 = t[i*n+k+5]/sum_t;
                float pt6 = p[i*n+k+5]-q6;
                float y_diff6 = y[i*d+j]-y[(k+5)*d+j+d];
                float grad_add6 = pt6 * y_diff6;
                grad_add6 *= t[i*n+k+5];
                grad_cy[i*d+j] += grad_add;
                grad_cy[i*d+j] += grad_add2;
                grad_cy[i*d+j] += grad_add3;
                grad_cy[i*d+j] += grad_add4;
                grad_cy[i*d+j] += grad_add5;
                grad_cy[i*d+j] += grad_add6;
            }
            grad_cy[i*d+j] *= eta;
            dy[i*d+j] *= alpha;
            dy[i*d+j] += grad_cy[i*d+j];
            y[i*d+j] += dy[i*d+j];
        }
    }
}

void gradientDescent_block(float* y, float* dy, float* grad_cy, float* p, float* t, float sum_t, int n, int d, float alpha, float eta, int NB1, int NB2) {
    eta *= -4;
    float q;
    for(int i=0; i<n; i+=NB1){
        for(int j=0; j<d; j+=NB2){
            for(int k=i+1; k<n; k+=NB1){
                for(int i1=i; i1<i+NB1; i1++){
                    for(int j1=j; j1<j+NB2; j1++){
                        for(int k1=k; k1<k+NB1; k1++){
                            q = t[i1*n+k1]/sum_t;
                            float pt = p[i1*n+k1]-q;
                            float y_diff = y[i1*d+j1]-y[k1*d+j1];
                            float grad_add = pt * y_diff;
                            grad_add *= t[i1*n+k1];
                            grad_cy[i1*d+j1] += grad_add;
                        }
                    }
                }
            }
        }
    }

    for(int i=0; i<n; i+=NB1){
        for(int j=0; j<d; j+=NB2){
            for(int i1=i; i1<i+NB1; i1++){
               for(int j1=j; j1<j+NB2; j1++){
                   grad_cy[i1*d+j1] *= eta;
                   dy[i1*d+j1] *= alpha;
                   dy[i1*d+j1] += grad_cy[i1*d+j1];
                   y[i1*d+j1] += dy[i1*d+j1];
               }
            }
        }
    }
}

void compute_t1(float* y, float* t, int n, int d) {
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

void compute_t2(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            float diff, dist = 0;
            for(int k=0; k<d; k++){
                diff = y[i*d+k] - y[j*d+k];
                dist += diff * diff;
            }
            t[i*n+j] = 1/(1+dist);
        }
    }
}

void compute_t3(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            float diff, dist = 1;
            for(int k=0; k<d; k++){
                diff = y[i*d+k] - y[j*d+k];
                dist += diff * diff;
            }
            t[i*n+j] = 1/dist;
        }
    }
}

void compute_t_scalar1(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            float diff1, diff2, dist = 1;
            for(int k=0; k<d; k+=2){
                diff1 = y[i*d+k] - y[j*d+k];
                diff2 = y[i*d+k+1] - y[j*d+k+1];
                diff1 = diff1 * diff1;
                diff2 = diff2 * diff2;
                dist += diff1;
                dist += diff2;
            }
            t[i*n+j] = 1/dist;
        }
    }
}

void compute_t_scalar2(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            float diff1, diff2, diff3, diff4, dist = 1;
            for(int k=0; k<d; k+=4){
                diff1 = y[i*d+k] - y[j*d+k];
                diff2 = y[i*d+k+1] - y[j*d+k+1];
                diff3 = y[i*d+k+2] - y[j*d+k+2];
                diff4 = y[i*d+k+3] - y[j*d+k+3];
                diff1 = diff1 * diff1;
                diff2 = diff2 * diff2;
                diff3 = diff3 * diff3;
                diff4 = diff4 * diff4;
                diff1 += diff2;
                diff3 += diff4;
                dist += diff1;
                dist += diff3;
            }
            t[i*n+j] = 1/dist;
        }
    }
}

void compute_t_scalar3(float* y, float* t, int n, int d) {
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            float diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, dist = 1;
            diff1 = y[i*d] - y[j*d];
            diff2 = y[i*d+1] - y[j*d+1];
            diff3 = y[i*d+2] - y[j*d+2];
            diff4 = y[i*d+3] - y[j*d+3];
            diff5 = y[i*d+4] - y[j*d+4];
            diff6 = y[i*d+5] - y[j*d+5];
            diff7 = y[i*d+6] - y[j*d+6];
            diff8 = y[i*d+7] - y[j*d+7];
            diff1 = diff1 * diff1;
            diff2 = diff2 * diff2;
            diff3 = diff3 * diff3;
            diff4 = diff4 * diff4;
            diff5 = diff5 * diff5;
            diff6 = diff6 * diff6;
            diff7 = diff7 * diff7;
            diff8 = diff8 * diff8;
            diff1 += diff5;
            diff2 += diff6;
            diff3 += diff7;
            diff4 += diff8;
            diff1 += diff3;
            diff2 += diff4;
            dist += diff1;
            dist += diff2;
            t[i*n+j] = 1/dist;
        }
    }
}

//set t to 1
void compute_t_block1(float* y, float* t, int n, int d, int NB1, int NB2) {
    float diff, dist;
    for(int i=0; i<n; i+=NB1){
        for(int j=i+1; j<n; j+=NB1){
            for(int k=0; k<d; k+=NB2){
                for(int i1=i; i1<i+NB1; i1+=1){
                    for(int j1=j; j1<j+NB1; j1+=1){
                        for(int k1=k; k1<k+NB2; k1+=1){
                            diff = y[i1*d+k1] - y[j1*d+k1];
                            dist = diff * diff;
                            t[i1*n+j1] += dist;
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<n; i+=NB1){
        for(int j=i+1; j<n; j+=NB1){
            for(int i1=i; i1<i+NB1; i1+=1){
                for(int j1=j; j1<j+NB1; j1+=1){
                    t[i1*n+j1] = 1/t[i1*n+j1];
                }
            }
        }
    }
}

//set t to 1
void compute_t_block_block1(float* y, float* t, int n, int d, int NB1, int NB2, int MU, int NU, int KU) {
    float diff, dist;
    for(int i=0; i<n; i+=NB1){
        for(int j=i+1; j<n; j+=NB1){
            for(int k=0; k<d; k+=NB2){
                for(int i1=i; i1<i+NB1; i1+=MU){
                    for(int j1=j; j1<j+NB1; j1+=NU){
                        for(int k1=k; k1<k+NB2; k1+=KU){
                            for(int i2=i1; i2<i1+MU; i2++){
                                for(int j2=j1; j2<j1+NU; j2++){
                                    for(int k2=k1; k2<k1+KU; k2++){
                                        diff = y[i2*d+k2] - y[j2*d+k2];
                                        dist = diff * diff;
                                        t[i2*n+j2] += dist;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for(int i=0; i<n; i+=NB1){
        for(int j=i+1; j<n; j+=NB1){
            for(int i1=i; i1<i+NB1; i1+=MU){
                for(int j1=j; j1<j+NB1; j1+=NU){
                    for(int i2=i1; i2<i1+MU; i2++){
                        for(int j2=j1; j2<j1+NU; j2++){
                            t[i2*n+j2] = 1/t[i2*n+j2];
                        }
                    }
                }
            }
        }
    }
}

float compute_sum_t1(float* t, int n) {
    float sum_t = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i!=j) sum_t += t[i*n+j];
        }
    }
    return sum_t;
}

float compute_sum_t2(float* t, int n) {
    float sum_t = 0;
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            sum_t += t[i*n+j];
        }
    }
    sum_t *= 2;
    return sum_t;
}

float compute_sum_t_scalar1(float* t, int n) {
    float sum_t = 0;
    float sum1;
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j+=2){
            sum1 = t[i*n+j] + t[i*n+j+1];
            sum_t += sum1;
        }
    }
    sum_t *= 2;
    return sum_t;
}

float compute_sum_t_scalar2(float* t, int n) {
    float sum_t = 0;
    float sum1, sum2;
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j+=4){
            sum1 = t[i*n+j] + t[i*n+j+1];
            sum2 = t[i*n+j+2] + t[i*n+j+3];
            sum1 += sum2;
            sum_t += sum1;
        }
    }
    sum_t *= 2;
    return sum_t;
}

float compute_sum_t_scalar3(float* t, int n) {
    float sum_t = 0;
    float sum1, sum2, sum3, sum4;
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j+=4){
            sum1 = t[i*n+j] + t[i*n+j+1];
            sum2 = t[i*n+j+2] + t[i*n+j+3];
            sum3 = t[i*n+j+4] + t[i*n+j+5];
            sum4 = t[i*n+j+6] + t[i*n+j+7];
            sum1 += sum3;
            sum2 += sum4;
            sum1 += sum2;
            sum_t += sum1;
        }
    }
    sum_t *= 2;
    return sum_t;
}

float compute_sum_t_block(float* t, int n, int NB) {
    float sum_t = 0;
    for(int i=0; i<n; i+=NB){
        for(int j=i+1; j<n; j+=NB){
            for(int i1=i; i1<i+NB; i1++){
                for(int j1=j; j1<j+NB; j1++){
                    sum_t += t[i1*n+j1];
                }
            }
        }
    }
    sum_t *= 2;
    return sum_t;
}

float compute_sum_t_block_block(float* t, int n, int NB, int MU, int NU) {
    float sum_t = 0;
    for(int i=0; i<n; i+=NB){
        for(int j=i+1; j<n; j+=NB){
            for(int i1=i; i1<i+NB; i1+=MU){
                for(int j1=j; j1<j+NB; j1+=NU){
                    for(int i2=i1; i2<i1+MU; i2++){
                        for(int j2=j1; j2<j1+NU; j2++){
                            sum_t += t[i2*n+j2];
                        }
                    }
                }
            }
        }
    }
    sum_t *= 2;
    return sum_t;
}

void getLowDimResult(float* y, float* y_trans, float* dy, float* grad_cy, float* p, float* t, int n_samples, int d_out, float alpha, float eta, int n_iter, int NB1, int NB2, int MU, int NU, int KU) {
    for (int i = 0; i < n_iter; i++) {
        //baseline
        //compute_t1(y, t, n_samples, d_out);
        //float sum_t = compute_sum_t1(t, n_samples);
        //gradientDescent1(y, dy, grad_cy, p, t, sum_t, n_samples, d_out, alpha, eta);

        //scalar
        //compute_t_scalar3(y, t, n_samples, d_out);
        //float sum_t = compute_sum_t_scalar3(t, n_samples);
        //gradientDescent_scalar3(y, dy, grad_cy, p, t, sum_t, n_samples, d_out, alpha, eta);

        //blocking
        compute_t_block1(y, t, n_samples, d_out, NB1, NB2);
        float sum_t = compute_sum_t_block(t, n_samples, NB1);
        gradientDescent_block(y, dy, grad_cy, p, t, sum_t, n_samples, d_out, alpha, eta, NB1, NB2);

	}
}
