import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from glob import glob


def list_pickles(folder):
    files = sorted(glob(os.path.join(folder, "*.pkl")))

    return files


def load_split(split_path):
    """Return X (n, d) and y (n,) as numpy arrays."""
    Xs, ys = [], []
    for f in list_pickles(split_path):
        with open(f, "rb") as fh:
            sample = pickle.load(fh)
        X = sample["X"]
        y = sample["y"]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)
        # flatten to 1D feature vector per sample
        X = X.reshape(-1)
        # ensure scalar int label
        y = int(y) if not np.isscalar(y) else int(y)
        Xs.append(X)
        ys.append(y)
    Xs = np.stack(Xs, axis=0)  # (n_samples, n_features)
    ys = np.asarray(ys, dtype=np.int64)
    return Xs, ys


if __name__ == '__main__':

    X_train, y_train = load_split(
        "/scratch/chntzi001/khula/processedBinLogReg/train")
    X_test,  y_test = load_split(
        "/scratch/chntzi001/khula/processedBinLogReg/test")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
