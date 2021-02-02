import numpy as np


def knn_predict_simple(X, y, x, k):
    distances = np.array([np.sqrt(np.sum(p)) for p in np.power(np.subtract(x, X), 2)])
    indecies = np.argsort(distances)
    # print(distances[indecies])
    # print(y[indecies])
    # voices = [y[indecies[i]] for i in range(min(k, len(distances)))]
    voices = y[indecies[0:k]]
    unique, frequency = np.unique(voices, return_counts=True)
    # print(list(zip(unique, frequency)))
    # voices = {}
    # for i in range(min(k, len(distances))):
    #     if np.int(y[indecies[i]]) in voices.keys():
    #         voices[np.int(y[indecies[i]])] += 1
    #     else:
    #         voices[np.int(y[indecies[i]])] = 1

    return list(zip(unique, frequency))


# X = [[2, 2], [1, 1], [2, 2], [100, 100]]
# y = [3, 1, 3, 100]
# x = [0.5, 0.5]
# k = 2
# print(knn_predict_simple(np.array(X), np.array(y), np.array(x), 3))
