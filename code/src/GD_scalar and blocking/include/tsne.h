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
void getLowDimResult(float* y, float* y_trans, float* dy, float* grad_cy, float* p, float* t, int n_samples, int d_out, float alpha, float eta, int n_iter, int NB1, int NB2, int MU, int NU, int KU);

/**
 * \brief compute squared euclidean distance
 *
 * @param X: data point
 * @param N: n_samples
 * @param D: input dimension
 * @param DD: output of Euclidean Distances
 */
void getSquaredEuclideanDistances(float* X, int n_samples, int d_in, float* DD);

#endif