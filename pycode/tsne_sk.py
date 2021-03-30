import numpy as np
from sklearn.manifold import TSNE

path_data = 'datasets/glove6B/glove_data_400kx300.txt'
path_embeddings = 'datasets/glove6B/glove_output_sk'
n = 1000
dims = range(1, 4)
random_seed = 0

for dim in dims:
	tsne = TSNE(n_components=dim, method='exact', random_state=random_seed)
	data = np.loadtxt(path_data, max_rows=n)
	embedding = tsne.fit_transform(data)
	np.savetxt(path_embeddings + '_' + str(n) + 'x' + str(dim) + '.txt', embedding, delimiter='\t')
