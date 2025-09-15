import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "Category": ["Developmentally delayed", "Developmentally typical"],
    "Cognitive": [30, 217],
    "Language": [55, 176],
    "Motor": [31, 171]
}
df = pd.DataFrame(data)
colors = sns.color_palette("muted", n_colors=3)


df_melt = df.melt(id_vars="Category", var_name="Subscale", value_name="Value")


ax = df_melt.pivot(index="Subscale", columns="Category", values="Value").plot(
    kind="bar", title="Sample Bar Graph", color=colors
)
ax.set_xticklabels(["Cognitive", "Language", "Motor"],
                   rotation=0,  fontsize=14, fontweight="bold")
ax.set_xlabel("")
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.legend()
plt.title("")

for container in ax.containers:
    ax.bar_label(container, label_type="edge", fontsize=9)

plt.savefig(f"bayley_dist.pdf",
            format="pdf", bbox_inches="tight")
