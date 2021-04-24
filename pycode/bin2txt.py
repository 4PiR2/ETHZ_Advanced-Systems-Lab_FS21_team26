import struct
import numpy as np

with open('datum/output', 'rb') as f:
	n, m = struct.unpack('i' * 2, f.read(4 * 2))
	data = struct.unpack('f' * n * m, f.read(4 * n * m))
mat = np.array(data).reshape((n, m))
np.savetxt('datasets/test/mnist_output_' + str(mat.shape[0]) + 'x' + str(mat.shape[1]) + '.txt', mat, delimiter='\t')
