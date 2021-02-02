import numpy as np


class Adam:
    '''Represents an Adam optimizer

    Fields:
        eta: learning rate
        beta1: first moment decay rate
        beta2: second moment decay rate
        epsilon: smoothing term
    '''

    eta: float
    beta1: float
    beta2: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        '''Initalizes `eta`, `beta1` and `beta2` fields'''
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

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
        m = 0
        v = 0
        for i in range(max_iter):
            grad = oracle.gradient(x)
            if np.linalg.norm(grad) < eps:
                break
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * np.power(grad, 2)

            m_cap = m / (1 - np.power(self.beta1, i + 1))
            v_cap = v / (1 - np.power(self.beta2, i + 1))

            x -= m_cap * self.eta / (np.sqrt(v_cap) + self.epsilon)
        return x
