import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from glob import glob
import argparse


def list_pickles(folder):
    # returns all the .pkl files
    files = sorted(glob(os.path.join(folder, "*.pkl")))
    return files


def get_score(task, y_dict, default=100):
    val = None

    if isinstance(y_dict, dict):  # gets value for the actual task: COG, LANG, MOT, age
        val = y_dict.get(task, None)

    if val is None or val == '':  # missing value
        return None
    if isinstance(val, (int, float)):  # converts all to ints
        return float(val)

    return float(val)


def load_sample(folder_path, task):
    # print("Loading:", folder_path)
    age = os.path.basename(folder_path).split('_')[3]      # '24'
    age_int = int(age)
    # print("Age int is: ", age_int)
    with open(folder_path, "rb") as fh:
        sample = pickle.load(fh)
    X, y = sample["X"], get_score(f"{age_int}M_AGE", sample.get("y", {}))
    if y is None:
        return None, None

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float32).reshape(-1)

    return X, y


def fit_scalers(train_files, task="COG", batch_size=256):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    n = 0
    for X_batch, y_batch in load_batch(train_files, batch_size=batch_size, task=task, shuffle=False):
        x_scaler.partial_fit(X_batch)
        y_scaler.partial_fit(y_batch.reshape(-1, 1))
        n += len(y_batch)
    print(f"saw {n} training samples")
    return x_scaler, y_scaler


def load_batch(files, batch_size=256, task="COG", shuffle=False, seed=0):
    id = np.arange(len(files))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(id)

    for i in range(0, len(files), batch_size):
        subset_batch = id[i:i+batch_size]
        Xs, ys = [], []
        for j in subset_batch:
            X, y = load_sample(files[j], task)
            if X is None or y is None:
                continue
            Xs.append(X)
            ys.append(y)
        yield np.vstack(Xs), np.asarray(ys, dtype=np.float32)


def train_model(train_files, x_scaler, y_scaler,
                total_epochs, batch_size, seed, task="COG"):

    model = SGDRegressor(
        loss="huber",
        epsilon=1.5,
        penalty="elasticnet",
        alpha=3e-3,
        l1_ratio=0.1,
        learning_rate="adaptive",
        eta0=3e-3,
        power_t=0.5,
        average=True,
        tol=None,
        max_iter=1,
        shuffle=False,
        warm_start=True,
        random_state=seed
    )
    print("\n")
    print("==== SGDRegressor hyperparameters ===")
    print(model.get_params(deep=False))

    files = np.array(train_files)

    for epoch in range(1, total_epochs+1):
        for X_batch, y_batch in load_batch(files, batch_size=batch_size, task=task,
                                           shuffle=True, seed=seed+epoch):
            X_batch_scaled = safe_standardise_X(X_batch, x_scaler, clip=5.0)
            y_batch_scaled = y_scaler.transform(y_batch.reshape(-1, 1)).ravel()
            model.partial_fit(X_batch_scaled, y_batch_scaled)
        print(f"Epoch {epoch}/{total_epochs} done")
    return model


def safe_standardise_X(X, x_scaler, clip=5.0):
    mean = x_scaler.mean_.astype(np.float32)
    std = x_scaler.scale_.astype(np.float32)
    std = np.where(std == 0.0, 1.0, std)
    Xs = (X - mean) / std
    Xs = np.clip(Xs, -clip, clip)   # helps with outliers
    return Xs


def evaluate_model(model, x_scaler, y_scaler,
                   test_files, batch_size, save_csv_path, task="COG"):
    maes, rmses, all_ytrue, all_ypred = [], [], [], []

    for X_test, y_true in load_batch(test_files, batch_size=batch_size, task=task, shuffle=False):
        X_test = safe_standardise_X(X_test, x_scaler, clip=5.0)
        y_pred_scaled = model.predict(X_test)
        # converts back to original data's scale
        y_pred = y_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).ravel()

        maes.append(mean_absolute_error(y_true, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        print(
            f"Test MAE={mean_absolute_error(y_true, y_pred):.6f}  RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.6f}", flush=True)
        all_ytrue.append(y_true)
        all_ypred.append(y_pred)
    all_ytrue = np.concatenate(all_ytrue) if all_ytrue else np.array([])
    all_ypred = np.concatenate(all_ypred) if all_ypred else np.array([])
    mae_global = mean_absolute_error(all_ytrue, all_ypred)
    rmse_global = np.sqrt(mean_squared_error(all_ytrue, all_ypred))
    r2 = r2_score(all_ytrue, all_ypred)
    print("=========================================================")
    print(
        f"OVERALL: Test MAE={mae_global:.6f}  RMSE={rmse_global:.6f}  R2={r2:.6f}", flush=True)

    mean_const = np.full_like(all_ytrue, all_ytrue.mean(), dtype=float)
    median_const = np.full_like(all_ytrue, np.median(all_ytrue), dtype=float)
    base_mean_mae = mean_absolute_error(all_ytrue, mean_const)
    base_mean_rmse = np.sqrt(mean_squared_error(all_ytrue, mean_const))
    base_med_mae = mean_absolute_error(all_ytrue, median_const)
    print(
        f"Based on baseline mean:   MAE={base_mean_mae:.3f}  RMSE={base_mean_rmse:.3f}", flush=True)
    print(f"Based on baseline median: MAE={base_med_mae: .3f}", flush=True)

    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    pd.DataFrame({"y_true": all_ytrue, "y_pred": all_ypred}
                 ).to_csv(save_csv_path, index=False)
    print(f"Saved predictions to: {save_csv_path}",  flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="LANG",
                   choices=["COG", "LANG", "MOT", "AGE"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_csv", type=str, required=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_dir = "/scratch/chntzi001/khula/processedLinReg/train"
    test_dir = "/scratch/chntzi001/khula/processedLinReg/test"
    save_csv = args.save_csv
    task = args.task
    batch_size = 256
    seed = args.seed
    epochs = 20

    # load pkl files
    train_files = list_pickles(train_dir)
    test_files = list_pickles(test_dir)

    ys = []  # just making a list of all the y values
    for X_batch, y_batch in load_batch(train_files, batch_size=batch_size, task=task, shuffle=True, seed=seed):
        ys.append(y_batch)
        if len(ys) > 5:
            break
    ys = np.concatenate(ys)
    print(
        f"CHECK: y stats: n={ys.size}, min={ys.min():.3f}, max={ys.max():.3f}, mean={ys.mean():.3f}, std={ys.std():.3f}", flush=True)

    x_scaler, y_scaler = fit_scalers(
        train_files, task=task, batch_size=batch_size)

    model = train_model(train_files, x_scaler, y_scaler,
                        total_epochs=epochs, batch_size=batch_size, seed=seed, task=task)

    # output check to see stats of score from training set
    train_slice = train_files[:1024]
    y_true, y_pred = [], []
    for X_batch, y_batch in load_batch(train_slice, batch_size=256, task=task, shuffle=False):
        y_pred_std = model.predict(x_scaler.transform(X_batch))
        y_pred_batch = y_scaler.inverse_transform(
            y_pred_std.reshape(-1, 1)).ravel()
        y_true.append(y_batch)
        y_pred.append(y_pred_batch)
    all_y_true = np.concatenate(y_true)
    all_y_pred = np.concatenate(y_pred)
    print("Checking train values: MAE=", mean_absolute_error(all_y_true, all_y_pred),
          "RMSE=", np.sqrt(mean_squared_error(all_y_true, all_y_pred)),
          "R2=", r2_score(all_y_true, all_y_pred))

    evaluate_model(model, x_scaler, y_scaler,
                   test_files, batch_size=batch_size, save_csv_path=save_csv, task=task)

# "/home/chntzi001/deepEEG/linearReg/preds.csv"
