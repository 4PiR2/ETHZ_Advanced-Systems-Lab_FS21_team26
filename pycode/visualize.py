import random
import numpy as np
import matplotlib.pyplot as plt

# path_embeddings = 'datasets/mnist/mnist_output_sk_1000x2.txt'
# path_labels = 'datasets/mnist/mnist_label_70kx1.txt'
path_embeddings = 'datasets/glove6B/glove_output_sk_1000x2.txt'
path_labels = 'datasets/glove6B/glove_label_400kx1.txt'
dimensions = None
# dimensions=[0]
n_samples = 100
path_plot = 'plots/plot.pdf'
show_plot = True
random_seed = 0
colors = None
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
font_size = 9

embeddings = np.loadtxt(path_embeddings)
labels = np.loadtxt(path_labels, dtype=str)
idx2label = np.unique(labels)
label2idx = {idx2label[i]: i for i in range(len(idx2label))}
n, d_out = embeddings.shape
index_samples = list(range(n))
random.seed(random_seed)
random.shuffle(index_samples)
if n_samples is None:
	n_samples = n
index_samples = index_samples[:n_samples]
if dimensions is None:
	dimensions = range(d_out)
else:
	d_out = len(dimensions)
embeddings = embeddings[index_samples][:, dimensions]
labels = labels[index_samples].reshape(len(index_samples))
lim_min = np.min(embeddings, axis=0) * 1.05
lim_max = np.max(embeddings, axis=0) * 1.1
if d_out == 1:
	embeddings = np.concatenate((embeddings, embeddings), axis=-1)
	ax = plt.figure().add_subplot()
	ax.set_xlim(lim_min[0], lim_max[0])
	ax.set_ylim(lim_min[0], lim_max[0])
elif d_out == 2:
	ax = plt.figure().add_subplot()
	ax.set_xlim(lim_min[0], lim_max[0])
	ax.set_ylim(lim_min[1], lim_max[1])
elif d_out == 3:
	ax = plt.figure().add_subplot(projection='3d')
	ax.set_xlim(lim_min[0], lim_max[0])
	ax.set_ylim(lim_min[1], lim_max[1])
	ax.set_zlim(lim_min[2], lim_max[2])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
else:
	print('d_out is not in [1, 2, 3]!')
	exit(0)
for e, l in zip(embeddings, labels):
	ax.text(*e, l, fontsize=font_size, color=colors[label2idx[l]] if colors is not None else None)
if path_plot is not None:
	plt.savefig(path_plot, bbox_inches='tight', pad_inches=0.1)
if show_plot:
	plt.show()
