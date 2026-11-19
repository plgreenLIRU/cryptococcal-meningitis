from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bayes import threshold_line

PROJECT_ROOT = Path(__file__).resolve().parent
SIMS_DIR = PROJECT_ROOT / "sims"
FIGURES_DIR = PROJECT_ROOT / "figures" / "posteriors"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

gold_standard_path = SIMS_DIR / "population_samples_gold_standard.csv"
population_path = gold_standard_path if gold_standard_path.exists() else SIMS_DIR / "population_samples_temp.csv"
df = pd.read_csv(population_path)

plt.figure()

# Plot limits
xmax = 1.0
ymax = 1.0

# Threshold line
pd_range = np.linspace(0, xmax, 500)
pb_range = np.clip(threshold_line(pd_range), 0, ymax)

# Background shading
plt.fill_between(
    pd_range, 0, pb_range,
    color="green", alpha=0.3, zorder=0
)
plt.fill_between(
    pd_range, pb_range, ymax,
    color="red", alpha=0.3, zorder=0
)

# Copy the colormap so we can modify it
cmap = plt.cm.Blues.copy()

# Make values below vmin (i.e. zero-count bins) transparent
cmap.set_under((1, 1, 1, 0))

# 2D histogram
plt.hist2d(
    df["pd"],
    df["pb"],
    bins=50,
    range=[[0, xmax], [0, ymax]],
    cmap=cmap,
    vmin=0.5,          # Zero-count bins are "under"
    zorder=1
)

# Threshold line
plt.plot(pd_range, pb_range, color="black", linewidth=2, zorder=2)

plt.colorbar(label="Count")
plt.xlabel(r"$p_d$")
plt.ylabel(r"$p_b$")
plt.xlim(0, xmax)
plt.ylim(0, ymax)

plt.tight_layout()
for extension in ("png", "pdf"):
    plt.savefig(FIGURES_DIR / f"population_posterior.{extension}", dpi=300, bbox_inches="tight")
plt.close()

# Standalone stasis threshold, with no population posterior overlay
fig, ax = plt.subplots()
ax.fill_between(pd_range, 0, pb_range, facecolor="white", edgecolor="black", hatch="..", zorder=0)
ax.fill_between(pd_range, pb_range, ymax, facecolor="white", edgecolor="black", hatch="/", zorder=0)
ax.plot(pd_range, pb_range, color="black", linewidth=2, linestyle="-", label=r"$p_b = \frac{p_d}{1-p_d}$", zorder=2)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_xlabel(r"$p_d$")
ax.set_ylabel(r"$p_b$")
ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)

fig.tight_layout()
for extension in ("png", "pdf"):
    fig.savefig(FIGURES_DIR / f"threshold_only.{extension}", dpi=600, bbox_inches="tight")
plt.close(fig)
