import numpy as np


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.dot(p, p)


a = [0, 1, 2, 2, 4, 1]
print(gini(a))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return np.sum(probs * (1.0 - probs))


print(gini(a))
