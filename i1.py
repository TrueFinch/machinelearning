# import collections
import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.dot(p, np.log2(p))
