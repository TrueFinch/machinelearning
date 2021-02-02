class Oracle:
    pass


import numpy as np


class GDM:
    '''Represents a Gradient Descent with Momentum optimizer

    Fields:
        eta: learning rate
        alpha: exponential decay factor
    '''

    eta: float
    alpha: float

    def __init__(self, *, alpha: float = 0.9, eta: float = 0.1):
        '''Initalizes `eta` and `aplha` fields'''
        self.eta = eta
        self.alpha = alpha

    def optimize(self, oracle, x0: np.ndarray, *,
                 max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
        '''Optimizes a function specified as `oracle` starting from point `x0`.
        The optimizations stops when `max_iter` iterations were completed or
        the L2-norm of the gradient at current point is less than `eps`

        Args:
            oracle: function to optimize
            x0: point to start from
            max_iter: maximal number of iterations
            eps: threshold for L2-norm of gradient

        Returns:
            A point at which the optimization stopped
        '''

        x = x0
        velocity = 0
        for _ in range(max_iter):
            grad = oracle.gradient(x)
            velocity = self.alpha * velocity - self.eta * grad
            if np.linalg.norm(grad) < eps:
                break
            x += velocity
        return x


class Oracle:
    def get_func(self, x):
        return x ** 2 - 10

    def get_grad(self, x):
        return 2 * x
