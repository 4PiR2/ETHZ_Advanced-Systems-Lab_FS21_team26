import numpy as np

mat_target = np.loadtxt('../output/p_tar.txt')
mat_test = np.loadtxt('../output/p_mut.txt')
diff = np.abs(mat_test - mat_target) / (np.abs(mat_target) + 1e-12)
print(np.max(diff[mat_target != 0]))
print(np.max(diff[diff < 1]))
pass
