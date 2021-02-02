import numpy as np
import deap
import sys
import matplotlib.pyplot as plt

from m1 import *


class GA:
    MAX_STEP = 1000
    POP_SIZE = 100
    SEL_SIZE = 50
    MAX_MUTATIONS_NUMBER = 10
    K_POINT = 100

    def __init__(self, task_comp: np.ndarray, task_time: np.ndarray, devs_coef: np.array):
        self.task_comp = task_comp
        self.task_time = task_time
        self.devs_coef = devs_coef
        self.population = np.array([])
        self.fitness_cache = np.array([])
        self.epoch = 0
        self.best_time = float("inf")
        self.best_member = np.array([])

    def initialize(self):
        self.population = np.array([np.random.randint(low=1, high=len(self.devs_coef) + 1, size=len(self.task_comp))
                                    for _ in range(self.POP_SIZE)])
        # print("start population:", self.population)

    def fitness(self, member):
        return self.calculateMaxDevTime(member)

    def selection(self):
        candidates_indecies = np.random.randint(low=0, high=self.POP_SIZE, size=np.min([self.SEL_SIZE, self.POP_SIZE]))

        candidates_time = np.zeros((len(candidates_indecies), len(self.population[0]) + 1), dtype="object")
        candidates_time[:, :-1] = self.population[candidates_indecies]
        candidates_time[:, -1] = self.fitness_cache[candidates_indecies]
        # print(candidates_time)
        sorted_candidates_time = candidates_time[candidates_time[:, -1].argsort()]
        # print(sorted_candidates_time)
        # print(sorted_candidates_time[:, :-1][:2])

        return sorted_candidates_time[:, :-1][:2]

    def crossover(self, a: np.ndarray, b: np.ndarray):
        if self.K_POINT == 1:
            point = np.random.randint(low=0, high=len(a))
            return single_point_crossover(a, b, point)
        if self.K_POINT == 2:
            point1 = np.random.randint(low=0, high=len(a) // 2)
            point2 = np.random.randint(low=point1 + 1, high=len(a))
            return two_point_crossover(a, b, point1, point2)
        else:
            points = np.sort(np.random.choice(np.arange(0, len(a)), replace=False, size=self.K_POINT))
            return k_point_crossover(a, b, points)

    def mutation(self, member):
        # print(f"Before mutation: {member}")
        index = np.random.randint(low=0, high=len(member))
        mutation = np.random.randint(low=1, high=len(self.devs_coef) + 1)
        member[index] = mutation
        # print(f"After mutation: {member}")
        return member

    def calculateMaxDevTime(self, member):
        t_max = -float("inf")
        for dev in range(1, len(self.devs_coef) + 1):
            if dev in member:
                indices = np.where(member == dev)[0]
                # print(member)
                # print(indices)
                # print(self.task_time[indices])
                t_max = np.max([t_max, np.dot(self.task_time[indices],
                                              self.devs_coef[dev - 1][self.task_comp[indices] - 1])])
        return t_max

    def step(self):
        self.epoch += 1
        self.fitness_cache = np.array(list(map(self.fitness, self.population)))
        loc_best_time_index = np.argmin(self.fitness_cache)
        if self.fitness_cache[loc_best_time_index] < self.best_time:
            self.best_time = self.fitness_cache[loc_best_time_index]
            self.best_member = self.population[np.argmin(self.fitness_cache)]
        print(f"Epoch: {self.epoch}, best_time={self.best_time}")
        # print(f"Best member: {self.population[np.argmin(self.fitness_cache)]}")
        next_population = np.array([self.crossover(*self.selection())[0] for _ in range(self.POP_SIZE)])

        members_to_mutate = np.random.randint(low=0, high=self.POP_SIZE,
                                              size=np.random.randint(low=0, high=self.MAX_MUTATIONS_NUMBER))
        if len(members_to_mutate) != 0:
            for index in members_to_mutate:
                next_population[index] = self.mutation(next_population[index])
        self.population = next_population


if __name__ == "__main__":
    n = int(input())
    comp = np.array(list(map(int, input().strip().split())))
    time = np.array(list(map(float, input().strip().split())))

    m = int(input())
    coef = np.array(list(map(lambda line: list(map(float, line.strip().split())), sys.stdin.readlines())))
    ga = GA(comp, time, coef)
    ga.initialize()
    times = []
    for i in range(ga.MAX_STEP):
        ga.step()
        times.append(ga.best_time)

    print(f"Best member: {ga.best_member}")
    # print(f"Score: {(1e6 / ga.best_time * ga.best_time)}")
    # print(ga.calculateMaxDevTime(np.array([1, 1, 1]))) # test
