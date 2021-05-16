#ifndef TSNE_H

/**
 * \brief compute symmetric affinities
 * 
 * \param X: data points
 * \param n_samples: number of data items (row of X)
 * \param d_in: input dimensions (col of X)
 * \param perp: perplexity input
 * \param P: symmetric affinities
 * \param squaredEuclideanDistances: Uninitialized Euclidean Distances Matrix
 */
void getSymmetricAffinity(float* X, int n_samples, int d_in, float perp, float* P, float* squaredEuclideanDistances);


/**
 * \brief use gradient descent to compute the result
 * 
 * \param affinity: symmetric affinities
 * \param n: number of data items
 * \param d: output dimention
 * \param y: result
 */
void getLowDimResult(float* y, float* dy, float* grad_cy, float* p, float* t, int n, int d, float alpha, float eta, int n_iter);

/**
 * \brief compute squared euclidean distance
 *
 * @param X
 * @param N
 * @param D
 * @param DD
 */
void getSquaredEuclideanDistances(float* X, int n_samples, int d_in, float* DD);

#endif