import numpy as np


def sus(fitness: np.ndarray, n: int, start: float) -> list:
    """Selects exactly `n` indices of `fitness` using Stochastic universal sampling alpgorithm.

    Args:
        fitness: one-dimensional array, fitness values of the population, sorted in descending order
        n: number of individuals to keep
        start: minimal cumulative fitness value

    Return:
        Indices of the new population"""

    def rws(sum_fitness, points):
        keep = []
        for p in points:
            i = 0
            while sum_fitness[i] < p:
                i += 1
            keep.append(i)
        return keep

    distance = np.sum(fitness) / n
    pointers = np.array([start + i * distance for i in range(n)])

    return rws(np.cumsum(fitness), pointers)


# fitness = np.array([10, 4, 3, 2, 1])
# print(*fitness[sus(fitness, 3, 6)])
