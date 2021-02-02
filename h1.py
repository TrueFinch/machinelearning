import numpy as np


def linear_func(theta, x):
    return np.dot(x, theta)


def linear_func_all(theta, m_x):
    return np.array([linear_func(theta, x) for x in m_x])


def mean_squared_error(theta, m_x, y):
    return np.sum(np.power(linear_func_all(theta, m_x) - y, 2)) / len(m_x)


def grad_mean_squared_error(theta, m_x, y):
    return 2 * np.dot(linear_func_all(theta, m_x) - y, m_x) / len(m_x)


def adam(X, y, *, max_iter=1500, eps=1e-8):
    eta = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    x = np.zeros(len(X[0]))
    m = np.zeros(len(X[0]))
    v = np.zeros(len(X[0]))
    for i in range(max_iter):
        grad = grad_mean_squared_error(x, X, y)
        if np.linalg.norm(grad) < eps:
            break
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * np.square(grad)

        m_cap = m / (1.0 - np.power(beta1, i + 1))
        v_cap = v / (1.0 - np.power(beta2, i + 1))

        x -= m_cap * eta / (np.sqrt(v_cap) + epsilon)
    return x


# def fit_linear_regression(X, y):
#     return adam(X, y)
import numpy as np


def fit_linear_regression(X, y):
    return np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.dot(y, X))


X = np.array([[1, 2], [3, 4], [4, 5]])

# theta = np.array([5, 6])

y = np.array([1, 2, 1])

print(fit_linear_regression(X, y))


def fit_linear_regression2(X, y):
    # def linear_func_all(theta, X):
    #     return np.dot(X, theta)
    #
    # def mean_squared_error(theta, X, y):
    #     return (1.0 / len(X)) * np.sum(np.square(y - linear_func_all(theta, X)))
    #
    # def grad_mean_squared_error(theta, X, y):
    #     return (-2.0 / len(X)) * np.dot(np.transpose(X), y - linear_func_all(theta, X))

    epoch = 0
    eta = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    eps = 1e-5
    max_iter = 1e3 + 500
    dim = len(X[0])
    point = np.zeros(dim)
    velocity = np.zeros(dim)
    momentum = np.zeros(dim)
    while epoch < max_iter:
        grad = grad_mean_squared_error(point, X, y)
        momentum = beta1 * momentum + (1.0 - beta1) * grad
        velocity = beta2 * velocity + (1.0 - beta2) * np.square(grad)
        if np.linalg.norm(grad) < eps:
            break
        corrected_momentum = momentum / (1.0 - np.power(beta1, epoch + 1))
        corrected_velocity = velocity / (1.0 - np.power(beta2, epoch + 1))
        point = point - (eta * corrected_momentum) / (np.sqrt(corrected_velocity) + epsilon)
        epoch += 1
    return point


print(fit_linear_regression2(X, y))
