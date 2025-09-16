""" Generates histograms for model predictions vs true labels"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSVPATH = "/Users/cccohen/logs/4-MOT-ft/predictions.csv"

df = pd.read_csv(CSVPATH, usecols=["Epoch", "Prediction", "True Label"])
df = df[df["Epoch"] == 39]
print(df)

preds = df["Prediction"].to_numpy()
true = df["True Label"].to_numpy()

bins = np.histogram_bin_edges(true, bins=20)

fig, ax = plt.subplots()

ax.hist(true, bins=bins,
        alpha=0.5, label='True Label', color='red',  edgecolor='black')


unique_preds = np.unique(preds)
print(unique_preds)

for value in unique_preds:
    pred_value = value
    pred_count = preds.size
    bin_width = bins[1] - bins[0]

    ax.text(pred_value, pred_count + 760,
            f"{pred_value} (prediction)",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="blue")

    ax.bar(pred_value, pred_count, width=bin_width*0.9,
           color='blue', alpha=0.5, edgecolor='black',
           label=f'Prediction', zorder=3)
    ax.axvline(pred_value, color='blue',  # to emphasise where the prediction is, the width of the bar is exaggerated for visibility
               linestyle='--', linewidth=1, zorder=4)


fig.suptitle("Motor subtest", fontsize=16, fontweight="bold")
ax.legend()
ax.set_xlabel('Value', fontweight="bold",
              fontsize=16)
ax.set_ylabel('Frequency', fontweight="bold",

              fontsize=16)
plt.savefig(f"4-MOT-ft-hist.pdf",
            format="pdf", bbox_inches="tight")
