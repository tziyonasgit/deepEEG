""" 
Purpose: Aggregates test accuracies across multiple training runs logged in text files to a csv
        
Author: Tziyona Cohen, UCT
"""
import os
import json
import numpy as np
import csv

save_dir = "/Users/cccohen/deepEEG/"
log_dir = "/Users/cccohen/logs/4-bin-age-scratch"
log_files = [f for f in os.listdir(
    log_dir) if f.endswith(".txt")]

print("log_files:", log_files)

# Collect accuracies per epoch across runs
test_acc_by_epoch = {}
test_acc_by_epoch.setdefault(0, []).append(0)
for file in log_files:
    print(file)
    file_path = os.path.join(log_dir, file)
    with open(file_path, "r") as infile:
        for line in infile:

            entry = json.loads(line)
            if "epoch" in entry and "test_accuracy" in entry:
                epoch = entry["epoch"] + 1
                acc = entry["test_accuracy"]

                if entry["epoch"] == 19:  # this one is defined for experiments trained for 20 epochs
                    print(entry["epoch"])
                    print(acc)
                if acc is None or (isinstance(acc, float) and np.isnan(acc)):
                    continue
                test_acc_by_epoch.setdefault(epoch, []).append(acc)

# Compute mean, std and count per epoch
stats_by_epoch = {}
for epoch, accs in test_acc_by_epoch.items():
    accs = np.asarray(accs, dtype=float)
    n = accs.size
    mean = float(np.mean(accs)) if n else float("nan")
    std = float(np.std(accs, ddof=1)) if n > 1 else 0.0  # 0 if only one run
    stats_by_epoch[epoch] = (mean, std, n)

# Save to CSV
csv_path = os.path.join(save_dir, "4-bin-age-scratch.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "mean_test_accuracy",
                    "std_test_accuracy", "n_runs"])
    for epoch in sorted(stats_by_epoch.keys()):
        mean, std, n = stats_by_epoch[epoch]
        writer.writerow([epoch, f"{mean:.6f}", f"{std:.6f}", n])

print(f"Saved mean/std to {csv_path}")
