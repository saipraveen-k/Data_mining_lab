# Data Mining Lab

Lab programs for the Data Mining course. Each script is self-contained (sample data is built in) and prints its results step by step.

| Program | What it covers |
|---|---|
| `Datapreprocessing.py` | Missing-value imputation, duplicates, IQR outliers, label/one-hot encoding, min-max & z-score scaling, binning |
| `ETL.py` | Extract → Transform → Load of sales records into a SQLite warehouse |
| `Similarity and Dissimilarity.py` | Euclidean, Manhattan, Minkowski, Chebyshev, cosine, SMC, Jaccard, Hamming, nominal dissimilarity |
| `AssociationRules.py` | Apriori algorithm from scratch with support, confidence and lift |
| `DescionTree.py` | ID3 decision tree from scratch (Play Tennis dataset) |
| `DecisionTreeClassifier.py` | scikit-learn decision tree on Iris with metrics and tree plot |
| `KMeans.py` | K-Means from scratch plus the elbow method |
| `Hierarchical.py` | Agglomerative clustering (single / complete / average linkage) with dendrograms |
| `Divisive.py` | Divisive clustering (DIANA) from scratch |

## Running

```bash
pip install -r requirements.txt
python AssociationRules.py
python "Similarity and Dissimilarity.py"
```

Scripts that draw plots save them as PNG files in the current folder.
