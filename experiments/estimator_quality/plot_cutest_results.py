import sys
import numpy as np 
import glob
import os

import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']

l_labels = ['2', 'd/3', 'd/2', '4/5 d', 'd']

base_path = sys.argv[1]

results = {dir_type : [[] for _ in range(len(l_labels))] for dir_type in direction_type}

directories = os.listdir(base_path)

for filename in directories:
    if not filename.endswith(".csv"):
        continue
    data = pd.read_csv(base_path +"/" + filename)
    d = data['dimension'].values[0]
    for dir_type in direction_type:
        num_directions = [2, d//3, d//2, round(d * (4/5)), d]
        for (i, l) in enumerate(num_directions):
            data_l = data[data['l'] == l]
            results[dir_type][i].append(np.mean(data_l[data_l['directions'] == dir_type]['err'].values))


fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes = axes.flatten()

for i in range(len(l_labels)):
    if i < len(l_labels) - 1:
        taus = np.logspace(-1, 1, 5000)
    else:
        taus = np.logspace(-7, 1, 5000)

    axes[i].set_title(f"$\\ell = {l_labels[i]}$", fontsize=18)
    for (j, dir_type) in enumerate(direction_type):
        result = np.array(results[dir_type][i]) 
        y = [np.sum(result < tau) / result.shape[0] for tau in taus]
        axes[i].plot(taus, y ,'-', lw=3, c=colors[j], label=dir_type.replace("_"," "), rasterized=True)
    axes[i].set_xscale("log")
    axes[i].set_xlabel("$\\tau$", fontsize=18)
axes[0].legend(loc='upper left')
axes[0].set_ylabel("$\\rho_{grad}(\\tau)$", fontsize=18)
fig.tight_layout()
fig.savefig("./cutest_benchmark_frac_solved.pdf", bbox_inches='tight')
plt.close(fig)