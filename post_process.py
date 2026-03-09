import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bayes import threshold_line

df = pd.read_csv("population_samples_gold_standard.csv")

plt.figure()
plt.hist2d(df["pd"], df["pb"], bins=50, cmap="Blues", vmin=0)

pd_range = np.linspace(0, 0.999, 100)
pb_range = threshold_line(pd_range)

plt.plot(pd_range, pb_range, color='black')
plt.colorbar()
plt.xlabel(r"$p_d$")
plt.ylabel(r"$p_b$")
plt.xlim([0, 0.5])
plt.ylim([0, 1])

plt.show()