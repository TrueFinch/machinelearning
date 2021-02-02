import numpy as np
from typing import Tuple


def single_point_crossover(a: np.ndarray, b: np.ndarray, point: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs single point crossover of `a` and `b` using `point` as crossover point.
    Chromosomes to the right of the `point` are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        point: crossover point

    Return:
        Two np.ndarray objects -- the offspring"""

    left = np.arange(len(a)) <= point
    right = np.arange(len(a)) > point

    return np.concatenate((a[left], b[right]), axis=0), np.concatenate((b[left], a[right]), axis=0)


def two_point_crossover(a: np.ndarray, b: np.ndarray, first: int, second: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs two point crossover of `a` and `b` using `first` and `second` as crossover points.
    Chromosomes between `first` and `second` are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        first: first crossover point
        second: second crossover point

    Return:
        Two np.ndarray objects -- the offspring"""
    left = np.arange(len(a)) <= first
    btw = np.logical_not(np.logical_xor(first < np.arange(len(a)), np.arange(len(a)) < second))
    right = second <= np.arange(len(a))
    return np.concatenate((a[left], b[btw], a[right]), axis=0), np.concatenate((b[left], a[btw], b[right]), axis=0)


def k_point_crossover(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs k point crossover of `a` and `b` using `points` as crossover points.
    Chromosomes between each even pair of points are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        points: one-dimensional array, crossover points

    Return:
        Two np.ndarray objects -- the offspring"""
    k = len(a)
    res_a = np.array([], dtype='i')
    res_b = np.array([], dtype='i')
    points = np.append(np.array([0]), points)
    if len(points) % 2 == 0:
        points = np.append(points, [k])
    for p in range(1, len(points), 2):
        range1 = np.logical_not(np.logical_xor(points[p - 1] <= np.arange(k), np.arange(k) <= points[p]))
        res_a = np.concatenate((res_a, a[range1]), axis=0)
        res_b = np.concatenate((res_b, b[range1]), axis=0)
        range2 = np.logical_not(np.logical_xor(points[p] < np.arange(k), np.arange(k) < points[p + 1]))
        res_a = np.concatenate((res_a, b[range2]), axis=0)
        res_b = np.concatenate((res_b, a[range2]), axis=0)

    if len(points) % 2 != 0:
        right = np.arange(k) >= points[len(points) - 1]
        res_a = np.concatenate((res_a, a[right]), axis=0)
        res_b = np.concatenate((res_b, b[right]), axis=0)

    return res_a, res_b


# a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# b = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
# prep = lambda x: ' '.join(map(str, x))
# print(*map(prep, single_point_crossover(a, b, 4)), '', sep='\n')
# print(*map(prep, two_point_crossover(a, b, 2, 7)), '', sep='\n')
# print(*map(prep, k_point_crossover(a, b, np.array([1, 5, 8]))), '', sep='\n')
# print(*map(prep, k_point_crossover(a, b, np.array([1, 5]))), '', sep='\n')
