import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("/Users/cccohen/predictions.csv")  # EPOCH, PRED, STD

# Compute upper/lower
df["Upper"] = df["PRED"] + df["STD"]
df["Lower"] = df["PRED"] - df["STD"]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(df["EPOCH"], df["PRED"], color="red", label="Predicted")
plt.fill_between(df["EPOCH"], df["Lower"], df["Upper"],
                 color="red", alpha=0.3, label="Predicted ± std")

plt.xlabel("Epoch")
plt.ylabel("Prediction ± std")
plt.title("Prediction across Epochs with Std Shading")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()
