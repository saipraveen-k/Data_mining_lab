"""
ID3 Decision Tree implemented from scratch (no ML libraries).

Uses the classic "Play Tennis" dataset. At each node the attribute with the
highest information gain is chosen, and the tree is built recursively.
"""
import math
from collections import Counter

ATTRIBUTES = ["Outlook", "Temperature", "Humidity", "Wind"]
DATA = [
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny", "Mild", "High", "Weak", "No"],
    ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Strong", "No"],
]


def entropy(rows):
    counts = Counter(r[-1] for r in rows)
    total = len(rows)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def information_gain(rows, attr_index):
    total = len(rows)
    remainder = 0.0
    for value in {r[attr_index] for r in rows}:
        subset = [r for r in rows if r[attr_index] == value]
        remainder += len(subset) / total * entropy(subset)
    return entropy(rows) - remainder


def build_tree(rows, attrs):
    labels = [r[-1] for r in rows]
    if len(set(labels)) == 1:
        return labels[0]
    if not attrs:
        return Counter(labels).most_common(1)[0][0]
    gains = {a: information_gain(rows, ATTRIBUTES.index(a)) for a in attrs}
    best = max(gains, key=gains.get)
    idx = ATTRIBUTES.index(best)
    tree = {best: {}}
    for value in sorted({r[idx] for r in rows}):
        subset = [r for r in rows if r[idx] == value]
        tree[best][value] = build_tree(subset, [a for a in attrs if a != best])
    return tree


def predict(tree, sample):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        tree = tree[attr].get(sample[attr], "Unknown")
    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + tree)
        return
    attr = next(iter(tree))
    for value, subtree in tree[attr].items():
        print(f"{indent}{attr} = {value}")
        print_tree(subtree, indent + "    ")


if __name__ == "__main__":
    print(f"Entropy of full dataset: {entropy(DATA):.3f}")
    for a in ATTRIBUTES:
        print(f"  Gain({a}) = {information_gain(DATA, ATTRIBUTES.index(a)):.3f}")

    tree = build_tree(DATA, ATTRIBUTES)
    print("\nID3 Decision Tree:")
    print_tree(tree)

    sample = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
    print(f"\nPrediction for {sample}: {predict(tree, sample)}")
