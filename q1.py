import numpy as np
import sys


def a(index, X, Y):
    x = X[index]
    indices = np.where(Y[index] != Y)[0]
    y = X[indices[np.argmin(np.linalg.norm(X[indices] - X[index], axis=1))]]

    indices = np.where(Y[index] == Y)[0]
    x_hat = X[indices[np.argmin(np.linalg.norm(X[indices] - y, axis=1))]]
    return np.linalg.norm(x_hat - y) / np.linalg.norm(x - y)


input() # drop first line

data = np.array(list(map(lambda line: list(map(float, line.strip().split())), sys.stdin.readlines())))
points = data[:, :data.shape[1] - 1]
labels = np.array(list(map(int, data[:, data.shape[1] - 1])))

for i in range(points.shape[0]):
    print(f"{a(i, points, labels):0.3f}", end=' ')
