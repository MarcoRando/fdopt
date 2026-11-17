import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from plot_common import COLOR, LABEL, MARKER, ORDER  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "grid.alpha": 0.5})

WANTED = ["spherical", "gaussian", "rademacher", "coordinate", "householder",
          "butterfly", "permuted_householder", "qr"]
direction_type = [f for f in ORDER if f in WANTED]          # keep the global ordering

l_labels = ["2", "d/3", "d/2", "4d/5", "d"]

base_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else "./cutest_benchmark_frac_solved.pdf"

results = {t: [[] for _ in l_labels] for t in direction_type}

for filename in sorted(os.listdir(base_path)):
    if not filename.endswith(".csv"):
        continue
    data = pd.read_csv(os.path.join(base_path, filename))
    d = data["dimension"].values[0]
    num_directions = [2, d // 3, d // 2, round(d * (4 / 5)), d]
    for t in direction_type:
        for i, l in enumerate(num_directions):
            data_l = data[data["l"] == l]
            results[t][i].append(np.mean(data_l[data_l["directions"] == t]["err"].values))

fig, axes = plt.subplots(1, len(l_labels), figsize=(16, 4))

for i, ax in enumerate(axes):
    taus = np.logspace(-1, 1, 5000) if i < len(l_labels) - 1 else np.logspace(-7, 1, 5000)
    ax.set_title(rf"$\ell = {l_labels[i]}$")
    for t in direction_type:
        result = np.sort(np.array(results[t][i]))
        y = np.searchsorted(result, taus, side="left") / result.shape[0]   # fraction with err < tau
        ax.plot(taus, y, "-", lw=3, c=COLOR[t], rasterized=True)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlabel(r"$\tau$", fontsize=16)
    if i == 0:
        ax.set_ylabel(r"$\rho_{grad}(\tau)$", fontsize=16, labelpad=8)

handles = [Line2D([0], [0], color=COLOR[t], lw=3, label=LABEL[t]) for t in direction_type]
ncol = min(len(handles), 5)
#fig.tight_layout()
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=ncol,
           frameon=False, handlelength=2.5, fontsize=14, mode='expand', borderaxespad=0.2)
#fig.subplots_adjust(bottom=0.32)
fig.tight_layout(rect=(0.0, 0.15, 1.0,1.0))
fig.savefig(out_path, bbox_inches="tight")
fig.savefig(os.path.splitext(out_path)[0] + ".png", bbox_inches="tight")
plt.close(fig)
