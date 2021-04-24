import numpy as np

mat_target = np.loadtxt('t2.txt')
mat_test = np.loadtxt('t1.txt')
diff = np.abs((mat_target - mat_test) / mat_target)
print(np.max(diff[mat_target != 0]))
print(np.max(diff[diff < 1]))
