"""
Decision Tree Classifier using scikit-learn on the Iris dataset.

- Splits data into train/test sets
- Trains a tree using the entropy (information gain) criterion
- Reports accuracy, confusion matrix and classification report
- Prints the learned tree as text rules and saves a plot of it
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), "\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Learned rules:\n", export_text(model, feature_names=list(iris.feature_names)))

plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=120)
print("Tree diagram saved to decision_tree.png")
