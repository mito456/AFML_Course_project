# Shared figure style; figures are saved as PNG and PDF in outputs/results/figures/
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)

CAMERA, AUDIO, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
W = 4.8

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {name}")


# Asymmetric error bars from value and interval
def err(v, lo, hi):
    return np.vstack([np.asarray(v) - np.asarray(lo), np.asarray(hi) - np.asarray(v)])
