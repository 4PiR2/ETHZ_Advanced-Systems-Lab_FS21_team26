import numpy as np

n_samples = 100
d_out = 3
eta = 50
alpha = 0.8
rep = 3

p = np.loadtxt('../datasets/random/random_sym_1000x1000.txt', max_rows=n_samples)[:, :n_samples]
y = np.loadtxt('../datasets/random/random_normal_10x10000.txt', max_rows=d_out)[:, :n_samples].T
y = np.loadtxt('../datasets/random/random_sym_1000x1000.txt', max_rows=d_out)[:, :n_samples].T
u = np.zeros((n_samples, d_out))

for i in range(rep):
	t = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			tij = np.sum((y[i] - y[j]) ** 2)
			t[i][j] = 1 / (1 + tij)
	t_sum = np.sum(t) - np.sum(t.diagonal())

	# np.savetxt('../output/t_tar.txt', t, delimiter='\t')

	g = np.zeros((n_samples, d_out))
	for i in range(n_samples):
		for j in range(n_samples):
			g[i] += (p[i, j] - t[i, j] / t_sum) * (y[i] - y[j]) * t[i, j]
	g *= 4

	u = -eta * g + alpha * u
	y += u

np.savetxt('../output/y_tar.txt', y, delimiter='\t')

mat_target = y
mat_test = np.loadtxt('../output/output_matrix.txt')
# mat_test[mat_test == 0] = mat_test.T[mat_test == 0]
diff = np.abs(mat_test - mat_target) / (np.abs(mat_target) + 1e-12)
print(np.max(diff))
