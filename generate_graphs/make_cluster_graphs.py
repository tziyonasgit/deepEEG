""" Generates stacked bar charts of cluster compositions by age"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.ticker as mtick


CSVPATH = "/Users/cccohen/logs/4-multi-age-scratch/clusterdist.csv"

epoch = 19
time_length = 4
type = "multi"

df = pd.read_csv(CSVPATH, usecols=[
                 "Cluster", "3M", "6M", "12M", "24M", "Total"])

age_order = ["3M", "6M", "12M", "24M"]
age = {0: '3M', 1: '6M', 2: '12M', 3: '24M'}

palette = {
    "3M": "#800080",
    "6M": "#5959ED",
    "12M": "#000000",
    "24M": "#0ABA0A"
}
label_to_color = {0: '#800080',
                  1: '#5959ED', 2: '#000000', 3: '#0ABA0A'}
palette = {age[k]: label_to_color[k] for k in age}

if type == "multi":
    cluster_map = {f"{epoch}_0": "1", f"{epoch}_1": "2",
                   f"{epoch}_2": "3", f"{epoch}_3": "4"}
    df = df[df["Cluster"].isin(
        [f"{epoch}_0", f"{epoch}_1", f"{epoch}_2", f"{epoch}_3"])]
    order = ["1", "2", "3", "4"]
else:
    cluster_map = {f"{epoch}_0": "1", f"{epoch}_1": "2"}
    df = df[df["Cluster"].isin(
        [f"{epoch}_0", f"{epoch}_1"])]
    order = ["1", "2"]


# compute proportions
df_prop = df.copy()
df_prop[age_order] = df[age_order].div(df[age_order].sum(axis=1), axis=0)

# reshape
df_long = df_prop.melt(id_vars="Cluster",
                       value_vars=age_order,
                       var_name="Age",
                       value_name="Proportion")

sns.set(style="whitegrid")


df_pivot = df_long.pivot(index="Cluster", columns="Age", values="Proportion")
# keep fixed legend/color order
df_pivot = df_pivot.reindex(columns=age_order)
df_pivot = df_pivot.rename(index=cluster_map)

df_pivot = df_pivot.reindex(order)

fig, ax = plt.subplots(figsize=(6, 6))
xpos = np.arange(len(df_pivot))
bar_width = 0.8

for i, (clust, row) in enumerate(df_pivot.iterrows()):
    # sort segments for THIS bar: largest proportion first (bottom)
    # tie-break by your fixed age_order so colors stay consistent
    sorted_ages = sorted(row.index, key=lambda a: (
        row[a], -age_order.index(a)), reverse=True)

    bottom = 0.0
    for a in sorted_ages:
        val = float(row[a])
        if val > 0:
            ax.bar(xpos[i], val, width=bar_width, bottom=bottom,
                   color=palette[a], edgecolor="white", linewidth=0.5)
            bottom += val

# axes labels
ax.set_ylabel("Proportion", fontsize=16, fontweight="bold")
ax.set_xlabel("Cluster", fontsize=16, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(df_pivot.index, rotation=0,  fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# legend (fixed order 3M,6M,12M,24M)
handles = [Patch(facecolor=palette[a], edgecolor="white", label=a)
           for a in age_order]

# totals above each bar
totals = (
    df.set_index("Cluster")["Total"]
      .rename(index=cluster_map)
      .reindex(df_pivot.index)
      .fillna(0)
)
for i, total in enumerate(totals):
    ax.text(i, 1.02, f"n={int(total)}",
            ha='center', va='bottom', fontsize=14, fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.margins(y=0.1)
plt.tight_layout()
# plt.title("Multi-age classification (4s samples)",
#           fontsize=16, fontweight="bold", pad=40, loc="left")
plt.savefig(f"{time_length}-{type}-age-scratch-e{epoch}.pdf",
            format="pdf", bbox_inches="tight")
