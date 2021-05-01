#ifndef TSNE_H
/**
 * \brief compute symmetric affinity
 * 
 * \param x: processed data read from file
 * \param n: row of x (number of data items)
 * \param d: col of x
 * \param perp: perplexity input
 * \param affinity: write back the return value here
 */
void getSymmetricAffinity(float* x, int n, int d, int perp, float* affinity);


/**
 * \brief use gradient descent to compute the result
 * 
 * \param affinity: symetric affinity
 * \param n: number of data items
 * \param d: output dimention
 * \param y: result
 */
void getLowDimResult(float* y, float* dy, float* grad_cy, float* p, float* t, int n, int d, int alpha, int eta, int n_iter);

/**
 * \brief compute squared euclidean distance
 *
 * @param X
 * @param N
 * @param D
 * @param DD
 */
void getSquaredEuclideanDistances(float* X, int N, int D, float* DD);

#endif