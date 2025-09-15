import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---- File paths ----
CSV_4S_FT = "/Users/cccohen/deepEEG/figures/4-bin-age-ft.csv"
CSV_4S_SCR = "/Users/cccohen/deepEEG/figures/4-bin-age-scratch.csv"
CSV_10S_FT = "/Users/cccohen/deepEEG/figures/10-bin-age-ft.csv"
CSV_10S_SCR = "/Users/cccohen/deepEEG/figures/10-bin-age-scratch.csv"

# ---- Style/legend constants ----
COLOR_PRE = "blue"
COLOR_SCR = "orange"
ALPHA_PRE_BAND = 0.18
ALPHA_SCR_BAND = 0.40
BASELINE_Y = 0.50  # multi-class chance level (e.g., 4 classes)


def load(path):
    """
    Load a CSV with columns: epoch, mean_test_accuracy, std_test_accuracy
    - Coerces to numeric, drops NaNs, sorts by epoch
    - Returns x, m, s, lo, hi (lo/hi clipped to [0,1])
    """
    try:
        df = pd.read_csv(
            path, usecols=["epoch", "mean_test_accuracy", "std_test_accuracy"])
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["mean_test_accuracy"] = pd.to_numeric(
        df["mean_test_accuracy"], errors="coerce")
    df["std_test_accuracy"] = pd.to_numeric(
        df["std_test_accuracy"],  errors="coerce")
    df = df.dropna(subset=["epoch", "mean_test_accuracy",
                   "std_test_accuracy"]).sort_values("epoch")

    if df.empty:
        print(f"[WARN] No valid rows after cleaning: {path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    x = df["epoch"].to_numpy(dtype=float)
    m = df["mean_test_accuracy"].to_numpy(dtype=float)
    s = df["std_test_accuracy"].to_numpy(dtype=float)
    lo = np.clip(m - s, 0.0, 1.0)
    hi = np.clip(m + s, 0.0, 1.0)
    return x, m, s, lo, hi


def panel(ax, title, series):
    """
    Plot multiple series on a single axes.
    series: list of dicts with keys: x, m, lo, hi, ls, lw, alpha, color
    """
    plotted_any = False
    xmins, xmaxs = [], []

    for d in series:
        x = np.asarray(d.get("x", []),  dtype=float)
        m = np.asarray(d.get("m", []),  dtype=float)
        lo = np.asarray(d.get("lo", []), dtype=float)
        hi = np.asarray(d.get("hi", []), dtype=float)

        finite = np.isfinite(x) & np.isfinite(
            m) & np.isfinite(lo) & np.isfinite(hi)
        if not finite.any():
            continue

        x, m, lo, hi = x[finite], m[finite], lo[finite], hi[finite]

        ax.plot(
            x, m,
            lw=d.get("lw", 2),
            ls=d.get("ls", "-"),
            label="_nolegend_",        # avoid auto-legend entries
            color=d.get("color", None),
        )
        ax.fill_between(
            x, lo, hi,
            alpha=d.get("alpha", 0.15),
            linewidth=0,
            color=d.get("color", None),
            label="_nolegend_",        # avoid auto-legend entries
        )

        xmins.append(x.min())
        xmaxs.append(x.max())
        plotted_any = True

    # baseline
    ax.axhline(BASELINE_Y, lw=2, ls=":", color="red", alpha=1)

    # titles/labels
    ax.set_title(title)
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=16)
    ax.set_ylabel("Mean test accuracy", fontweight="bold", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    # x-limits & ticks only if at least one series plotted
    if plotted_any:
        x_min, x_max = float(min(xmins)), float(max(xmaxs))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max >= x_min:
            ax.set_xlim(x_min, x_max)
            xrng = int(round(x_max - x_min))
            step = 5 if xrng >= 5 else max(1, xrng or 1)
            ax.set_xticks(np.arange(int(x_min), int(x_max) + 1, step))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    # ---- 4-second panel ----
    x4, m4, s4, lo4, hi4 = load(CSV_4S_FT)
    xs, ms, ss, los, his = load(CSV_4S_SCR)
    series_4s = [
        dict(x=x4, m=m4, lo=lo4, hi=hi4, ls="-",  lw=2.0,
             alpha=ALPHA_PRE_BAND, color=COLOR_PRE),
        dict(x=xs, m=ms, lo=los, hi=his, ls="--", lw=1.8,
             alpha=ALPHA_SCR_BAND, color=COLOR_SCR),
    ]
    panel(ax1, "4-second", series_4s)
    ax1.set_title("4-second samples", fontweight="bold", fontsize=16, pad=20)

    # ---- 10-second panel ----
    x10, m10, s10, lo10, hi10 = load(CSV_10S_FT)
    xs, ms, ss, los, his = load(CSV_10S_SCR)
    series_10s = [
        dict(x=x10, m=m10, lo=lo10, hi=hi10, ls="-",
             lw=2.0, alpha=ALPHA_PRE_BAND, color=COLOR_PRE),
        dict(x=xs,  m=ms,  lo=los,  hi=his,  ls="--",
             lw=1.8, alpha=ALPHA_SCR_BAND, color=COLOR_SCR),
    ]
    panel(ax2, "10-second", series_10s)
    ax2.set_title("10-second samples", fontweight="bold", fontsize=16, pad=20)

    # ----- custom horizontal legend (centered below) with BOTH std bands -----
    legend_handles = [
        Line2D([0], [0], color=COLOR_PRE, lw=2.0,
               ls='-',  label='Pretrained (mean)'),
        Patch(facecolor=COLOR_PRE, alpha=ALPHA_PRE_BAND,
              label='Pretrained ±1 std'),
        Line2D([0], [0], color=COLOR_SCR, lw=1.8,
               ls='--', label='Scratch (mean)'),
        Patch(facecolor=COLOR_SCR, alpha=ALPHA_SCR_BAND, label='Scratch ±1 std'),
        Line2D([0], [0], color='red', lw=2, ls=':',
               label=f'Baseline ({BASELINE_Y:.2f})'),
    ]
    # fig.legend(
    #     handles=legend_handles,
    #     loc="lower center",
    #     ncol=5,
    #     frameon=False,
    #     handlelength=3.0,
    #     columnspacing=1.4,
    #     borderaxespad=0.8,
    #     prop={"size": 14, "weight": "bold"},   # <-- bigger, bold legend text
    # )

    # leave bottom space for the legend
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    plt.savefig("bin-age-ft.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
