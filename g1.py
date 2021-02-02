import numpy as np


def linear_func(theta, x):
    return np.dot(x, theta)


def linear_func_all(theta, m_x):
    return np.array([linear_func(theta, x) for x in m_x])


def mean_squared_error(theta, m_x, y):
    return np.sum(np.power(linear_func_all(theta, m_x) - y, 2)) / len(m_x)


def grad_mean_squared_error(theta, m_x, y):
    return 2 * np.dot(linear_func_all(theta, m_x) - y, m_x) / len(m_x)


X = np.array([[1, 2], [3, 4], [4, 5]])

theta = np.array([5, 6])

y = np.array([1, 2, 1])

print(linear_func_all(theta, X))  # --> array([17, 39, 50])

print(mean_squared_error(theta, X, y))  # --> 1342.0

print(grad_mean_squared_error(theta, X, y))  # --> array([215.33333333, 283.33333333])
