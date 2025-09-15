import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob


LOGPATH = "/Users/cccohen/logs/multi-age-ft"


def newgraph(set, metric):  # set = val/test, metric = loss/acc
    save_dir = "/Users/cccohen/deepEEG/"
    log_dirs = glob.glob(
        '/Users/cccohen/logs/multi-age-ft/run*/')
    log_files = []

    for d in log_dirs:
        print("d: ", d)
        for f in os.listdir(d):
            if f.endswith(".txt") and "log" in f:
                log_files.append((d, f))

    print("log_files: ", log_files)

    metric_key = {
        ("test", "loss"): "test_loss",
        ("test", "accuracy"): "test_accuracy",
    }

    metric_column = {
        ("test", "loss"): "TestLoss",
        ("test", "accuracy"): "TestAccuracy",
    }

    key = (set, metric)
    if key not in metric_key:
        print(f"Invalid set/metric combination: {set}/{metric}")
        return

    json_key = metric_key[key]
    col_name = metric_column[key]

    all_records = []
    new_labels = []

    for dir_path, log_file in log_files:
        run_name = log_file.replace(".txt", "")
        lr = None
        bs = None
        best_val_acc = None
        best_test_acc = None
        print(run_name)
        log_file_path = os.path.join(dir_path, log_file)
        print("the log_file is: ", log_file_path)
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Extract hyperparams from final config line
                    if 'learning_rate' in data:
                        lr = data['learning_rate']
                    if 'batch_size' in data:
                        bs = data['batch_size']

                    if 'epoch' in data and json_key in data:
                        all_records.append({
                            'Epoch': data['epoch'],
                            col_name: data[json_key],
                            'Run': run_name,
                            'lr': lr,
                            'batch_size': bs
                        })

                    if 'epoch' in data:
                        best_test_acc = data['max_accuracy_test']
                        best_epoch = data['epoch']

                except json.JSONDecodeError:
                    print(
                        f"⚠️ Skipping invalid JSON line in {log_file}: {line[:50]}")
                    continue

        label = f"lr:{lr} batch size:{bs}"
        new_labels.append(label)
    # Build DataFrame
    df = pd.DataFrame(all_records)

    if df.empty:
        print("No valid epoch data found.")
        return

    # Melt to long format for Seaborn
    df_melted = df.melt(id_vars=['Epoch', 'Run', 'lr', 'batch_size'],
                        value_vars=[col_name],
                        var_name='Metric', value_name='Value')

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_melted, x='Epoch',
                      y='Value', hue='Run', style=None, markers=True, dashes=False, errorbar=None)

    title = f"{set} {metric}"
    plt.title(f"{title} over epochs (all runs)")
    plt.xlabel(f"Epoch")
    plt.ylabel(title)

    plt.tight_layout()
    ax.legend(title="Legend", labels=new_labels)
    image_name = f"{set}_{metric}.png"
    image_name = os.path.join(save_dir, image_name)
    plt.savefig(image_name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    sets = ["test", "val"]
    metrics = ["loss", "accuracy"]
    for set in sets:
        for metric in metrics:
            newgraph(set, metric)

    print("done!!!!!!!")
