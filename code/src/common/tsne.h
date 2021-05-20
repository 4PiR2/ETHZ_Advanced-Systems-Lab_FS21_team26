#ifndef TSNE_H

/**
 * \brief compute symmetric affinities
 * 
 * \param x: data points
 * \param n_samples: number of data items (row of x)
 * \param d_in: input dimensions (col of x)
 * \param perplexity: perplexity input
 * \param p: symmetric affinities
 * \param DD: Uninitialized Euclidean Distances Matrix
 */
void getSymmetricAffinity(float *x, int n_samples, int d_in, float perplexity, float *p, float *DD);


/**
 * \brief use gradient descent to compute the result
 * 
 * \param affinity: symmetric affinities
 * \param n: number of data items
 * \param d: output dimention
 * \param y: result
 */
void getLowDimResult(float *y, float *dy, float *grad_cy, float *p, float *t, int n, int d, float alpha, float eta,
                     int n_iter);


#endif