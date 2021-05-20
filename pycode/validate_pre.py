import numpy as np

n_samples = 500
d_in = 784

x = np.loadtxt('../datasets/mnist/mnist_data_70kx784.txt', max_rows=n_samples)[:, :d_in]

p = np.zeros((n_samples, n_samples))
for i in range(n_samples):
	for j in range(n_samples):
		p[i][j] = np.sum((x[i] - x[j]) ** 2)

perp_tar = 50
# beta := -1 / (2 * sigma^2)
beta_l = 0
beta_r = 0
beta_last = 0
e_l = np.zeros(n_samples)
e_m = np.zeros(n_samples)
e_r = np.zeros(n_samples)
epsilon = 1e-7

count = 0
betas = np.zeros(n_samples)

h_tar = np.log(perp_tar)
for i in range(n_samples):
	beta = -1 / np.max(p[i])
	beta_l_ub = True
	beta_r_ub = True
	e_m = np.exp(p[i] * beta)
	while beta != beta_last:
		beta_last = beta
		count += 1
		e_m = np.exp(p[i] * beta)
		s = np.sum(e_m) - 1
		h = -np.sum(e_m * p[i] * beta) / s + np.log(s)
		prep = np.exp(h)
		if h > h_tar + epsilon:
			beta_l = beta
			beta_l_ub = False
			e_l = e_m
			if beta_r_ub:
				beta *= 2
				e_m = e_m ** 2
			else:
				beta = (beta_l + beta_r) / 2
				e_m = (e_l * e_r) ** 0.5
		elif h < h_tar - epsilon:
			beta_r = beta
			beta_r_ub = False
			e_r = e_m
			if beta_l_ub:
				beta /= 2
				e_m = e_m ** 0.5
			else:
				beta = (beta_l + beta) / 2
				e_m = (e_l * e_r) ** 0.5
		else:
			break
	p[i] = e_m / s
	p[i, i] = 0
	betas[i] = beta

count /= n_samples

p = (p + p.T) / (2 * n_samples)

mat_target = p
mat_test = np.loadtxt('output_matrix.txt')
diff = np.abs(mat_test - mat_target) / (np.abs(mat_target) + 1e-12)
print(np.max(diff[mat_target != 0]))
