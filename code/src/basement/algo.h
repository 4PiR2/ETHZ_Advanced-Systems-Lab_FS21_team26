#ifndef ALGO_H
#define ALGO_H

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// Compute squared Euclidean distance matrix
void computeSquaredEuclideanDistance(double *X, int N, int D, double *DD) {
	const double *XnD = X;
	for (int n = 0; n < N; ++n, XnD += D) {
		const double *XmD = XnD + D;
		double *curr_elem = &DD[n * N + n];
		*curr_elem = 0.0;
		double *curr_elem_sym = curr_elem + N;
		for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
			*(++curr_elem) = 0.0;
			for (int d = 0; d < D; ++d) {
				*curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
			}
			*curr_elem_sym = *curr_elem;
		}
	}
}

// Makes data zero-mean
void zeroMean(double *X, int N, int D) {
	// Compute data mean
	double *mean = (double *) calloc(D, sizeof(double));
	if (mean == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	int nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
		nD += D;
	}
	for (int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}
	// Subtract data mean
	nD = 0;
	for (int n = 0; n < N; n++) {
		for (int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
		nD += D;
	}
	free(mean);
	mean = NULL;
}

// Generates a Gaussian random number
double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while ((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Compute input similarities with a fixed perplexity
void computeGaussianPerplexity(double *X, int N, int D, double *P, double perplexity) {
	// Compute the squared Euclidean distance matrix
	double *DD = (double *) malloc(N * N * sizeof(double));
	if (DD == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	computeSquaredEuclideanDistance(X, N, D, DD);
	// Compute the Gaussian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {
		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta = DBL_MAX;
		double tol = 1e-5;
		double sum_P;
		// Iterate until we found a good perplexity
		int iter = 0;
		while (!found && iter < 200) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;
			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for (int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);
			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if (Hdiff < tol && -Hdiff < tol) {
				found = true;
			} else {
				if (Hdiff > 0) {
					min_beta = beta;
					if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				} else {
					max_beta = beta;
					if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}
			// Update iteration counter
			iter++;
		}
		// Row normalize P
		for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}
	// Clean up memory
	free(DD);
	DD = NULL;
}

// Compute gradient of the t-SNE cost function (exact)
void computeExactGradient(double *P, double *Y, int N, int D, double *dC) {
	// Make sure the current gradient contains zeros
	for (int i = 0; i < N * D; i++) dC[i] = 0.0;
	// Compute the squared Euclidean distance matrix
	double *DD = (double *) malloc(N * N * sizeof(double));
	if (DD == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	computeSquaredEuclideanDistance(Y, N, D, DD);
	// Compute Q-matrix and normalization sum
	double *Q = (double *) malloc(N * N * sizeof(double));
	if (Q == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	double sum_Q = .0;
	int nN = 0;
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < N; m++) {
			if (n != m) {
				Q[nN + m] = 1 / (1 + DD[nN + m]);
				sum_Q += Q[nN + m];
			}
		}
		nN += N;
	}
	// Perform the computation of the gradient
	nN = 0;
	int nD = 0;
	for (int n = 0; n < N; n++) {
		int mD = 0;
		for (int m = 0; m < N; m++) {
			if (n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for (int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}
	// Free memory
	free(DD);
	DD = NULL;
	free(Q);
	Q = NULL;
}

void run_tsne(double *X, int N, int D, double *Y, int no_dims, double perplexity, int rand_seed, int max_iter) {
	// Set random seed
	srand((unsigned int) rand_seed);
	// Set learning parameters
	double momentum = .8;
	double eta = 200.0;
	// Allocate some memory
	double *dY = (double *) malloc(N * no_dims * sizeof(double));
	double *uY = (double *) malloc(N * no_dims * sizeof(double));
	if (dY == NULL || uY == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	for (int i = 0; i < N * no_dims; i++) uY[i] = .0;
	// Normalize input data (to prevent numerical problems)
	zeroMean(X, N, D);
	double max_X = .0;
	for (int i = 0; i < N * D; i++) {
		if (fabs(X[i]) > max_X) max_X = fabs(X[i]);
	}
	for (int i = 0; i < N * D; i++) X[i] /= max_X;
	// Compute input similarities for exact t-SNE
	double *P;
	// Compute similarities
	P = (double *) malloc(N * N * sizeof(double));
	if (P == NULL) {
		printf("Memory allocation failed!\n");
		exit(1);
	}
	computeGaussianPerplexity(X, N, D, P, perplexity);
	// Symmetrize input similarities
	int nN = 0;
	for (int n = 0; n < N; n++) {
		int mN = (n + 1) * N;
		for (int m = n + 1; m < N; m++) {
			P[nN + m] += P[mN + n];
			P[mN + n] = P[nN + m];
			mN += N;
		}
		nN += N;
	}
	double sum_P = .0;
	for (int i = 0; i < N * N; i++) sum_P += P[i];
	for (int i = 0; i < N * N; i++) P[i] /= sum_P;
	// Initialize solution (randomly)
	for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
	// Perform main training loop
	for (int iter = 0; iter < max_iter; iter++) {
		// Compute (approximate) gradient
		computeExactGradient(P, Y, N, no_dims, dY);
		// Perform gradient update (with momentum and gains)
		for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * dY[i];
		for (int i = 0; i < N * no_dims; i++) Y[i] = Y[i] + uY[i];
		// Make solution zero-mean
		zeroMean(Y, N, no_dims);
	}
	// Clean up memory
	free(dY);
	free(uY);
	free(P);
}

#endif //ALGO_H
