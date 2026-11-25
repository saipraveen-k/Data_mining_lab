"""
K-Means clustering.

1. From-scratch implementation (random init, assign, update, repeat until stable).
2. Elbow method with scikit-learn to choose k.
3. Cluster plot saved to kmeans_clusters.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def kmeans(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for it in range(1, max_iter + 1):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = dists.argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                        for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    sse = ((X - centroids[labels]) ** 2).sum()
    return labels, centroids, sse, it


if __name__ == "__main__":
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=7)

    labels, centroids, sse, iters = kmeans(X, k=4)
    print(f"From-scratch K-Means converged in {iters} iterations, SSE = {sse:.2f}")
    print("Centroids:\n", centroids.round(3))

    print("\nElbow method (scikit-learn):")
    inertias = []
    for k in range(1, 9):
        inertia = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_
        inertias.append(inertia)
        print(f"  k={k}: inertia={inertia:.1f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(range(1, 9), inertias, "o-")
    a1.set(title="Elbow method", xlabel="k", ylabel="Inertia (SSE)")
    a2.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
    a2.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200)
    a2.set(title="K-Means clusters (k=4)")
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=120)
    print("\nPlots saved to kmeans_clusters.png")
