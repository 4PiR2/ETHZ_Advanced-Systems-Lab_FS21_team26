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
void getSymmetricAffinity(double* x, int n, int d, int perp, double* affinity);


/**
 * \brief use gradient descent to compute the result
 * 
 * \param affinity: symetric affinity
 * \param n: number of data items
 * \param d: output dimention
 * \param y: result
 */
void getLowdResult(double* affinity, int n, int d, double* y);

#endif