import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bayes import threshold_line

df = pd.read_csv("population_samples_gold_standard.csv")

plt.figure()

# Plot limits
xmax = 0.5
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

plt.show()
