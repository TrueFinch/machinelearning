import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.dot(p, np.log2(p))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.dot(p, p)


def var(x):
    return np.sum(np.square(x - np.mean(x))) / len(x)


def tree_split(X, y, criterion):
    criterion_func = None
    if criterion == 'var':
        criterion_func = var
    elif criterion == 'gini':
        criterion_func = gini
    elif criterion == 'entropy':
        criterion_func = entropy

    if criterion_func is None:
        return 0
    x_count = X.shape[0]
    f_count = X.shape[1]

    best_metric = float('inf')
    best_i = float('inf')
    best_j = float('inf')
    for i in range(f_count):
        for j in range(x_count):
            indices = X[:, i] <= X[j, i]
            l_features, r_features = y[indices], y[np.invert(indices)]
            metric = criterion_func(l_features) * len(l_features) + criterion_func(r_features) * len(r_features)
            if metric < best_metric:
                best_metric = metric
                best_i = i
                best_j = j
    return best_i, best_j
