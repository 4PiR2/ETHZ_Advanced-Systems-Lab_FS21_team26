import numpy as np

mat_target = np.loadtxt('../output/t1.txt')
mat_test = np.loadtxt('../output/t2.txt')
diff = np.abs(mat_test - mat_target) / (np.abs(mat_target) + 1e-12)
print(np.max(diff[mat_target != 0]))
print(np.max(diff[diff < 1]))
