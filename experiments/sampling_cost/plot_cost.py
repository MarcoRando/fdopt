import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly',  'qr']

dimensions = [50, 100, 500, 1000, 5000, 10000]

result_path = sys.argv[1]


data = pd.read_csv(result_path)


fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Computation Time for Direction Matrix Generation", fontsize=18)
axes = axes.flatten()
for (i, d) in enumerate(dimensions):
    ax = axes[i]
    d_data = data[data['d'] == d]
    for dir_type in direction_type:
        dir_data = d_data[d_data['directions'] == dir_type]
        mu_err = dir_data['mean_time'].values
        std_err = dir_data['std_time'].values

        ax.plot(dir_data['l'].values, mu_err,'o-', label=dir_type, rasterized=True)
        ax.fill_between(dir_data['l'].values, mu_err - std_err, mu_err + std_err, alpha=0.4, rasterized=True)
    ax.set_yscale('log')
    ax.set_xlabel('$\\ell$', fontsize=12)
    ax.set_title(f"$d = {d}$", fontsize=16)
    if i == 0:
        ax.legend(loc='upper left', fontsize=10)
    if i == 0 or i ==3:
        ax.set_ylabel("Time ($s$)", fontsize=12)
fig.tight_layout()
fig.savefig("./sampling_cost.pdf", bbox_inches='tight')
plt.close(fig)