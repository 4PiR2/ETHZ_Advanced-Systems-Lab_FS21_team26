import numpy as np
from sklearn.linear_model import LinearRegression

n_samples = 1e7
x = np.arange(-1, .1 / n_samples, 1 / n_samples)
X = np.stack([x ** 3, x ** 2, x]).T
y = np.exp(x)
reg = LinearRegression(fit_intercept=True).fit(X, y)
print(reg.score(X, y))
print(*reg.coef_, reg.intercept_)
