""" Plot of std vs accuracy for synthetic data experiment on classification task"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

std = [0.0001, 0.001, 0.01, 0.1, 1, 5, 6, 10, 50, 100]
acc = [1, 1, 1, 1, 1, 0.9694727105, 0.9146623497,
       0.617946346, 0.2523126735, 0.2405180389]

plt.figure(figsize=(7, 5))
plt.plot(std, acc, marker="o", linestyle="-", color="blue", label="Accuracy")

plt.xscale("log")
plt.axhline(y=0.25, color="red", linestyle="--",
            linewidth=1.5, label="Baseline (0.25)")

plt.xlabel("Standard deviation of added Gaussian noise", fontweight="bold",
           fontsize=16)
plt.ylabel("Accuracy", fontweight="bold",
           fontsize=16)
plt.title("Accuracy vs Noise STD", fontweight="bold",
          fontsize=16)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(f"synthetic_acc.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
