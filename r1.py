import numpy as np
import sys

n, m, k, t = list(map(int, input().strip().split()))

data = np.array(list(map(lambda line: list(map(float, line.strip().split())), sys.stdin.readlines())))

points = data[:, :data.shape[1] - 1]
clusters = np.array(list(map(int, data[:, data.shape[1] - 1])))

for step in range(t):
    # print("step number: ", step)
    means = np.array([np.mean(points[np.where(clusters == cluster_id)[0]], axis=0)
                      if cluster_id in clusters else np.zeros(m) for cluster_id in range(k)])

    old_clusters = np.copy(clusters)
    for index, point in enumerate(points):
        min_norm = float("inf")
        for cluster_id in range(k):
            tmp = np.linalg.norm(means[cluster_id] - point)
            if tmp < min_norm:
                clusters[index] = cluster_id
                min_norm = tmp

    if (old_clusters == clusters).all():
        break

for cluster_id in clusters:
    print(f"{cluster_id}")
