import numpy as np


class GradientOptimizer:
    def __init__(self, oracle, x0):
        self.oracle = oracle
        self.x0 = x0

    def optimize(self, iterations, eps, alpha):
        x = self.x0
        for _ in range(iterations):
            grad = self.oracle.get_grad(x)
            if np.linalg.norm(grad) < eps:
                break
            x -= alpha * grad
        return x


class Oracle:
    def get_func(self, x):
        return x ** 2 - 10

    def get_grad(self, x):
        return 2 * x


go = GradientOptimizer(Oracle(), 4)
print(go.optimize(100, 1e-7, 0.1))
