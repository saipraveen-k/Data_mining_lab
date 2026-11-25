"""
Measures of Similarity and Dissimilarity.

Numeric data : Euclidean, Manhattan, Minkowski (p=3), Chebyshev distances, cosine similarity
Binary data  : Simple Matching Coefficient, Jaccard coefficient, Hamming distance
Nominal data : simple mismatch dissimilarity d(i, j) = (p - m) / p
Mixed data   : full dissimilarity matrix for several objects
"""
import numpy as np


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    return np.sum(np.abs(x - y))


def minkowski(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def chebyshev(x, y):
    return np.max(np.abs(x - y))


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def binary_counts(a, b):
    f11 = np.sum((a == 1) & (b == 1))
    f00 = np.sum((a == 0) & (b == 0))
    f10 = np.sum((a == 1) & (b == 0))
    f01 = np.sum((a == 0) & (b == 1))
    return f11, f00, f10, f01


def smc(a, b):
    f11, f00, f10, f01 = binary_counts(a, b)
    return (f11 + f00) / (f11 + f00 + f10 + f01)


def jaccard(a, b):
    f11, _, f10, f01 = binary_counts(a, b)
    return f11 / (f11 + f10 + f01)


def nominal_dissimilarity(x, y):
    p = len(x)
    m = sum(1 for a, b in zip(x, y) if a == b)
    return (p - m) / p


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4], dtype=float)
    y = np.array([2, 4, 1, 3], dtype=float)
    print("Numeric vectors:", x, y)
    print(f"  Euclidean distance : {euclidean(x, y):.4f}")
    print(f"  Manhattan distance : {manhattan(x, y):.4f}")
    print(f"  Minkowski (p=3)    : {minkowski(x, y, 3):.4f}")
    print(f"  Chebyshev distance : {chebyshev(x, y):.4f}")
    print(f"  Cosine similarity  : {cosine_similarity(x, y):.4f}")

    a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    print("\nBinary vectors:", a, b)
    print(f"  SMC              : {smc(a, b):.4f}")
    print(f"  Jaccard          : {jaccard(a, b):.4f}")
    print(f"  Hamming distance : {np.sum(a != b)}")

    objects = [["red", "small", "round"], ["red", "large", "round"],
               ["blue", "small", "square"], ["blue", "large", "round"]]
    print("\nNominal dissimilarity matrix:")
    n = len(objects)
    M = np.array([[nominal_dissimilarity(objects[i], objects[j]) for j in range(n)] for i in range(n)])
    print(M.round(3))

    points = np.array([[1, 2], [3, 5], [2, 0], [4, 5]], dtype=float)
    print("\nEuclidean dissimilarity matrix:")
    print(np.array([[euclidean(p, q) for q in points] for p in points]).round(3))
