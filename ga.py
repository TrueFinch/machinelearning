import numpy as np
import deap
import sys
import matplotlib.pyplot as plt
import keyboard
from m1 import *

last_try_path = "last_try.txt"
output_path = "output.txt"

is_cont = True


class GA:
    MAX_STEP = 1000
    POP_SIZE = 10000
    SEL_SIZE = 200
    MIN_MUTATIONS_NUMBER = 20
    MAX_MUTATIONS_NUMBER = 50
    MAX_PERSONAL_MUTATIONS_NUMBER = 25
    K_POINT = 25

    def __init__(self, task_comp: np.ndarray, task_time: np.ndarray, devs_coef: np.array):
        self.task_comp = task_comp
        self.task_time = task_time
        self.devs_coef = devs_coef
        self.population = np.array([])
        self.fitness_cache = np.array([])
        self.epoch = 0
        self.best_time = float("inf")
        self.best_member = np.array([])

    def initialize(self, is_continue: bool = False):
        if is_continue:
            print("Continue last try")
            with open(last_try_path, "r") as fin:
                lines = list(map(lambda line: line.strip().split(), fin.readlines()))
                lines = list(map(lambda row: list(map(int, row)), lines))
                self.population = np.array(lines)
            return
        self.population = np.array([np.random.randint(low=1, high=len(self.devs_coef) + 1, size=len(self.task_comp))
                                    for _ in range(self.POP_SIZE)])
        # print("start population:", self.population)
        pass

    def fitness(self, member):
        return self.calculateMaxDevTime(member)

    def selection(self):
        candidates_indecies = np.random.choice(np.arange(0, self.POP_SIZE),
                                               replace=False,
                                               size=np.min([self.SEL_SIZE, self.POP_SIZE]))
        # candidates_indecies = np.random.randint(low=0, high=self.POP_SIZE, size=np.min([self.SEL_SIZE, self.POP_SIZE]))

        candidates_time = np.zeros((len(candidates_indecies), len(self.population[0]) + 1), dtype="object")
        candidates_time[:, :-1] = self.population[candidates_indecies]
        candidates_time[:, -1] = self.fitness_cache[candidates_indecies]
        sorted_candidates_time = candidates_time[candidates_time[:, -1].argsort()]

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
        index = np.random.randint(low=0, high=len(member), size=self.MAX_PERSONAL_MUTATIONS_NUMBER)
        mutation = np.random.randint(low=1, high=len(self.devs_coef) + 1, size=len(index))
        member[index] = mutation
        # print(f"After mutation: {member}")
        return member

    def calculateMaxDevTime(self, member):
        t_max = -float("inf")
        for dev in np.unique(member):
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
        print(f"Epoch: {self.epoch}", end="\t")
        print(f"best_time={self.best_time:.3f}", end="\t")
        print(f"local_best_time={self.fitness_cache[loc_best_time_index]:.3f}")
        # print(f"unique members={len(np.unique([tuple(row) for row in self.population], axis=1))}")
        # print(f"Best member: {self.population[np.argmin(self.fitness_cache)]}")
        next_population = np.array([self.crossover(*self.selection()) for _ in range(self.POP_SIZE // 2)])
        next_population = next_population.reshape((self.POP_SIZE, 1000))
        members_to_mutate = np.random.randint(low=0, high=self.POP_SIZE,
                                              size=np.random.randint(low=self.MIN_MUTATIONS_NUMBER,
                                                                     high=self.MAX_MUTATIONS_NUMBER))
        if len(members_to_mutate) != 0:
            for index in members_to_mutate:
                # print(f"Before mutation: {next_population[index]}")
                next_population[index] = self.mutation(next_population[index])
                # print(f"After mutation: {next_population[index]}")
        self.population = next_population
        pass


if __name__ == "__main__":
    n = int(input())
    comp = np.array(list(map(int, input().strip().split())))
    time = np.array(list(map(float, input().strip().split())))

    m = int(input())
    coef = np.array(list(map(lambda line: list(map(float, line.strip().split())), sys.stdin.readlines())))
    ga = GA(comp, time, coef)
    ga.initialize(is_cont)
    times = []
    stop = False


    def stop_func(a):
        global stop
        stop = True


    keyboard.on_press_key("`", stop_func)
    for i in range(ga.MAX_STEP):
        ga.step()
        times.append(ga.best_time)
        if stop:
            with open(last_try_path, "w") as fout:
                fout.write("\n".join([" ".join(list(map(str, member))) for member in ga.population]))
                fout.close()
            break
    with open(output_path, "w") as fout:
        fout.write(" ".join(list(map(str, ga.best_member))))
        fout.close()
    # print(f"Best member: {ga.best_member}")
    print(f"Score: {(1e5 / ga.best_time * 10.1216128)}")
    # print(ga.calculateMaxDevTime(np.array([1, 1, 1]))) # test
