import numpy as np


# def knn_predict_simple(X, y, x, k):
#     distances = np.array([np.sqrt(np.sum(p)) for p in np.power(np.subtract(x, X), 2)])
#     indecies = np.argsort(distances)
#     voices = y[indecies[0:k]]
#     unique, frequency = np.unique(voices, return_counts=True)
#     return list(zip(unique, frequency))
#
#
# def my_predict(X, y, x, k):
#     res = knn_predict_simple(X, y, x, k)
#     res = sorted(res, key=lambda pair: pair[1])
#     return res[0][0]


def create_subset(array, index_):
    a1 = array[:index_]
    a2 = array[index_ + 1:]
    return np.concatenate((a1, a2))


def loo_score(predict, X, y, k):
    error = 0
    for index, x in enumerate(X):
        predicted_y = predict(create_subset(X, index), create_subset(y, index), x, k)
        error += int(y[index] == predicted_y)
    return error


# X = np.array([[2, 2], [1, 1], [2, 2], [100, 100]])
# y = np.array([3, 1, 3, 100])
# # x = [0.5, 0.5]
# k = 2
# print(loo_score(my_predict, X, y, k))
# print(knn_predict_simple(np.array(X), np.array(y), np.array(x), 3))