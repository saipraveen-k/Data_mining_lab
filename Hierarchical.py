"""
Agglomerative Hierarchical Clustering.

1. A from-scratch single-linkage implementation that prints every merge step.
2. The same data clustered with SciPy (single, complete, average linkage) and a
   dendrogram saved to dendrogram.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

LABELS = ["A", "B", "C", "D", "E", "F"]
POINTS = np.array([[1, 1], [1.5, 1.5], [5, 5], [3, 4], [4, 4], [3, 3.5]])


def single_linkage(X, labels):
    clusters = [[i] for i in range(len(X))]
    D = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(-1))
    while len(clusters) > 1:
        best = (None, None, np.inf)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(D[i, j] for i in clusters[a] for j in clusters[b])
                if d < best[2]:
                    best = (a, b, d)
        a, b, d = best
        merged = clusters[a] + clusters[b]
        names = lambda c: "".join(labels[i] for i in sorted(c))
        print(f"merge {names(clusters[a]):>6} + {names(clusters[b]):<6} at distance {d:.3f}")
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [merged]


if __name__ == "__main__":
    print("From-scratch single linkage:")
    single_linkage(POINTS, LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, method in zip(axes, ["single", "complete", "average"]):
        Z = linkage(POINTS, method=method)
        print(f"\n{method} linkage, 2 clusters ->", dict(zip(LABELS, fcluster(Z, 2, "maxclust").tolist())))
        dendrogram(Z, labels=LABELS, ax=ax)
        ax.set_title(f"{method.title()} linkage")
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=120)
    print("\nDendrograms saved to dendrogram.png")
