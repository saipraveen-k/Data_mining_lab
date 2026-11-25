"""
Divisive Hierarchical Clustering (DIANA - DIvisive ANAlysis), implemented from scratch.

Starts with all points in one cluster and repeatedly splits the cluster with the
largest diameter:
  1. The point with the highest average dissimilarity starts a "splinter group".
  2. Points that are on average closer to the splinter group than to the rest move over.
  3. Repeat until the requested number of clusters is reached.
"""
import numpy as np

POINTS = np.array([
    [1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
    [3.5, 5.0], [4.5, 5.0], [3.5, 4.5], [9.0, 9.5], [8.5, 9.0],
])


def distance_matrix(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def diameter(cluster, D):
    if len(cluster) < 2:
        return 0.0
    return max(D[i, j] for i in cluster for j in cluster)


def split(cluster, D):
    remaining = list(cluster)
    avg = [np.mean([D[i, j] for j in remaining if j != i]) for i in remaining]
    splinter = [remaining.pop(int(np.argmax(avg)))]
    moved = True
    while moved and len(remaining) > 1:
        moved = False
        best_i, best_diff = None, 0.0
        for i in remaining:
            to_rest = np.mean([D[i, j] for j in remaining if j != i])
            to_splinter = np.mean([D[i, j] for j in splinter])
            if to_rest - to_splinter > best_diff:
                best_i, best_diff = i, to_rest - to_splinter
        if best_i is not None:
            remaining.remove(best_i)
            splinter.append(best_i)
            moved = True
    return sorted(splinter), sorted(remaining)


def diana(X, n_clusters):
    D = distance_matrix(X)
    clusters = [list(range(len(X)))]
    step = 0
    while len(clusters) < n_clusters:
        target = max(clusters, key=lambda c: diameter(c, D))
        clusters.remove(target)
        a, b = split(target, D)
        clusters.extend([a, b])
        step += 1
        print(f"Step {step}: split {sorted(target)} -> {a} | {b}")
    return clusters


if __name__ == "__main__":
    final = diana(POINTS, n_clusters=3)
    print("\nFinal clusters:")
    for k, c in enumerate(final, 1):
        print(f"  Cluster {k}: points {c} -> {POINTS[c].tolist()}")
