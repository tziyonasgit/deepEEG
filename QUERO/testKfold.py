from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# Create an imbalanced dataset: 90% class 0, 10% class 1
X, y = make_classification(n_samples=200, n_features=20, n_informative=5,
                           n_redundant=0, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")


def testStrat():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        print(" Train class distribution:", np.bincount(y[train_idx]))
        print(" Test class distribution: ", np.bincount(y[test_idx]))

        # Simple classifier
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        print(classification_report(y[test_idx], preds, digits=3))


def myStrat():

    root = "/Users/cccohen/deepEEG"

    abnormalFolder = os.path.join(root, "abTest")
    normalFolder = os.path.join(root, "normTest")
