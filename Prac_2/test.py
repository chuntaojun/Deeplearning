import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

np.random.seed(int(time.time()))
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0
y = np.dot(X, coef)
y += 0.01 * np.random.normal((n_samples,))

n_samples = X.shape[0]
X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test = X[n_samples / 2:], y[n_samples / 2:]

print(X_train.shape)
print(y_train.shape)
alpha = 0.1
lasso = Lasso(max_iter=10000, alpha=alpha)

lasso.fit(X_train, y_train)
