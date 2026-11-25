"""
Association Rule Mining using the Apriori algorithm (implemented from scratch).

Steps:
1. Find all frequent itemsets whose support >= min_support.
2. Generate rules X -> Y from each frequent itemset whose confidence >= min_confidence.
3. Report support, confidence and lift for every rule.
"""
from itertools import combinations

TRANSACTIONS = [
    {"milk", "bread", "butter"},
    {"bread", "butter"},
    {"milk", "bread"},
    {"milk", "eggs"},
    {"bread", "eggs", "butter"},
    {"milk", "bread", "butter", "eggs"},
    {"milk", "bread", "eggs"},
    {"bread", "butter"},
]


def support(itemset, transactions):
    """Fraction of transactions that contain every item in itemset."""
    return sum(1 for t in transactions if itemset <= t) / len(transactions)


def apriori(transactions, min_support):
    """Return {frozenset(itemset): support} for all frequent itemsets."""
    items = sorted({i for t in transactions for i in t})
    current = [frozenset([i]) for i in items]
    frequent = {}
    k = 1
    while current:
        level = {}
        for c in current:
            s = support(c, transactions)
            if s >= min_support:
                level[c] = s
        frequent.update(level)
        # Join step: build (k+1)-itemsets from frequent k-itemsets
        keys = list(level)
        k += 1
        candidates = set()
        for a, b in combinations(keys, 2):
            union = a | b
            if len(union) == k:
                # Prune step: every k-subset must itself be frequent
                if all(frozenset(sub) in level for sub in combinations(union, k - 1)):
                    candidates.add(union)
        current = list(candidates)
    return frequent


def generate_rules(frequent, min_confidence):
    rules = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                conf = sup / frequent[antecedent]
                if conf >= min_confidence:
                    lift = conf / frequent[consequent]
                    rules.append((antecedent, consequent, sup, conf, lift))
    return sorted(rules, key=lambda r: (-r[3], -r[4]))


def fmt(s):
    return "{" + ", ".join(sorted(s)) + "}"


if __name__ == "__main__":
    MIN_SUPPORT = 0.3
    MIN_CONFIDENCE = 0.6

    frequent = apriori(TRANSACTIONS, MIN_SUPPORT)
    print(f"Frequent itemsets (min_support={MIN_SUPPORT}):")
    for itemset, s in sorted(frequent.items(), key=lambda x: (len(x[0]), -x[1])):
        print(f"  {fmt(itemset):<28} support={s:.2f}")

    print(f"\nAssociation rules (min_confidence={MIN_CONFIDENCE}):")
    for a, c, s, conf, lift in generate_rules(frequent, MIN_CONFIDENCE):
        print(f"  {fmt(a):>22} -> {fmt(c):<18} support={s:.2f} confidence={conf:.2f} lift={lift:.2f}")
