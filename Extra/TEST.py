from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.zeros((10, 1))  # dummy features
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

skf = StratifiedKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"FOLD {fold + 1}")
    print("Test indices:", test_idx)
    print("Test labels:", np.array(y)[test_idx])
    print("Train labels:", np.array(y)[train_idx])
    print("---")
