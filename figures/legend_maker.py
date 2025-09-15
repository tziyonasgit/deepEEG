import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define your palette (thesis-friendly)
palette = {
    "3M": "#E69F00",   # orange
    "6M": "#56B4E9",   # blue
    "12M": "#009E73",  # green
    "24M": "#CC79A7"   # magenta
}

age_order = ["3M", "6M", "12M", "24M"]

# Create legend handles (just color + label)
handles = [Patch(facecolor=palette[age], edgecolor="white", label=age)
           for age in age_order]

# Make a blank figure just for the legend
fig, ax = plt.subplots(figsize=(4, 1))  # adjust size as needed
ax.axis("off")  # hide axes

# Legend with text formatting
legend = ax.legend(
    handles=handles,
    loc="center",
    ncol=len(age_order),   # all in one row
    frameon=False,
    fontsize=14,           # bigger labels
    title="Age",
    title_fontsize=16,     # bigger title
)

# Save only the legend
fig.savefig("legend.pdf", bbox_inches="tight")
plt.close(fig)
