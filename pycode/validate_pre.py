import numpy as np

n_samples = 500
d_in = 784

x = np.loadtxt('datasets/mnist/mnist_data_70kx784.txt', max_rows=n_samples)[:, :d_in]

p = np.zeros((n_samples, n_samples))
for i in range(n_samples):
	for j in range(n_samples):
		p[i][j] = np.sum((x[i] - x[j]) ** 2)

# p = np.loadtxt('datasets/random/random_1000x1000.txt', max_rows=n_samples)[:, :n_samples]

perp_tar = 50
sigma_l = 0
sigma_r = 0
epsilon = 1e-7

count = 0
betas = np.zeros(n_samples)

for i in range(n_samples):
	sigma = (np.max(p[i]) / 2) ** 0.5
	sigma_l_ub = True
	sigma_r_ub = True
	while True:
		count += 1
		pi = np.exp(-p[i] / (2 * sigma ** 2))
		pi[i] = 1
		pi /= np.sum(pi)
		hi = -np.sum(pi * np.log2(pi))
		perp = 2 ** hi
		if perp < perp_tar - epsilon:
			sigma_l = sigma
			sigma_l_ub = False
			if sigma_r_ub:
				sigma *= 2
			else:
				sigma = (sigma_l + sigma_r) / 2
		elif perp > perp_tar + epsilon:
			sigma_r = sigma
			sigma_r_ub = False
			if sigma_l_ub:
				sigma /= 2
			else:
				sigma = (sigma_l + sigma) / 2
		else:
			break
	p[i] = pi
	p[i, i] = 0
	beta = -1 / (2 * sigma ** 2)
	betas[i] = beta
	a = 0

count /= n_samples

p = (p + p.T) / (2 * n_samples)

mat_target = p
mat_test = np.loadtxt('output_matrix.txt')
# mat_test[mat_test == 0] = mat_test.T[mat_test == 0]
diff = np.abs((mat_target - mat_test) / mat_target)
print(np.max(diff[mat_target != 0]))
a = 0
