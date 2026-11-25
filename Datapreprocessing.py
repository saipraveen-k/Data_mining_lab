"""
Data Preprocessing

Demonstrates the standard preprocessing steps on a small, deliberately messy dataset:
1. Handling missing values (mean / mode imputation)
2. Removing duplicate rows
3. Detecting and capping outliers (IQR method)
4. Encoding categorical attributes (label + one-hot encoding)
5. Feature scaling (min-max normalisation and z-score standardisation)
6. Discretisation (equal-width binning)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

data = pd.DataFrame({
    "Age":     [25, 32, np.nan, 45, 29, 32, 120, 38, np.nan, 50],
    "Salary":  [40000, 52000, 61000, np.nan, 45000, 52000, 58000, 70000, 48000, 90000],
    "City":    ["Delhi", "Mumbai", "Delhi", np.nan, "Chennai", "Mumbai", "Delhi", "Chennai", "Mumbai", "Delhi"],
    "Purchased": ["No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes"],
})

print("1. Raw data\n", data, "\n")
print("Missing values per column:\n", data.isnull().sum(), "\n")

# 1. Missing values
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Salary"] = data["Salary"].fillna(data["Salary"].mean())
data["City"] = data["City"].fillna(data["City"].mode()[0])
print("2. After imputation\n", data, "\n")

# 2. Duplicates
before = len(data)
data = data.drop_duplicates().reset_index(drop=True)
print(f"3. Removed {before - len(data)} duplicate row(s)\n")

# 3. Outliers with the IQR rule
q1, q3 = data["Age"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = data[(data["Age"] < low) | (data["Age"] > high)]
print(f"4. Age outliers outside [{low:.1f}, {high:.1f}]:\n", outliers, "\n")
data["Age"] = data["Age"].clip(low, high)

# 4. Encoding
data["Purchased"] = LabelEncoder().fit_transform(data["Purchased"])
data = pd.get_dummies(data, columns=["City"], prefix="City", dtype=int)
print("5. After encoding\n", data, "\n")

# 5. Scaling
data["Age_minmax"] = MinMaxScaler().fit_transform(data[["Age"]]).ravel()
data["Salary_zscore"] = StandardScaler().fit_transform(data[["Salary"]]).ravel()
print("6. After scaling\n", data[["Age", "Age_minmax", "Salary", "Salary_zscore"]].round(3), "\n")

# 6. Discretisation
data["Age_group"] = pd.cut(data["Age"], bins=3, labels=["Young", "Middle", "Senior"])
print("7. Equal-width binning of Age\n", data[["Age", "Age_group"]])
