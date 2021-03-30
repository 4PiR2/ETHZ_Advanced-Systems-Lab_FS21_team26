import numpy as np

mat = np.random.random(size=(1000, 1000))
# mat = np.random.normal(scale=1e-4, size=(10, 10000))
# mat = (mat + mat.T) / 2
np.savetxt('datasets/random/random_' + str(mat.shape[0]) + 'x' + str(mat.shape[1]) + '.txt', mat, delimiter='\t')
