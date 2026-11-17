import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# same colours / labels / order as every other figure (see plot_common.py)
from plot_common import COLOR, LABEL, ORDER  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# same style as plot_grad_acc.py
try:
  plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
  plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "grid.alpha": 0.5})

base_dir = sys.argv[1]
reps = int(sys.argv[2])
table_path = sys.argv[3] if len(sys.argv) > 3 else "./frac_solved_table.tex"
data_dir = f"{base_dir}/ffd/cutest"

WANTED = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
direction_type = [f for f in ORDER if f in WANTED]          # keep the global ordering
num_dirs_label = ["2", "d/3", "d/2", "4d/5", "d"]

UNSTRUCTURED = [t for t in direction_type if t in ('gaussian', 'spherical', 'rademacher')]
STRUCTURED = [t for t in direction_type if t not in UNSTRUCTURED]
TABLE_TAUS = [1e-13, 1e-10, 1e-7, 1e-4]


def style_axes(axes, xlabel, ylabel):
  for i, ax in enumerate(axes):
    ax.set_title(rf"$\ell = {num_dirs_label[i]}$")
    ax.set_xlabel(xlabel, fontsize=16)
  axes[0].set_ylabel(ylabel, fontsize=16, labelpad=8)


def save_with_legend(fig, path):
  handles = [Line2D([0], [0], color=COLOR[t], lw=3, label=LABEL[t]) for t in direction_type]
  ncol = min(len(handles), 5)
  fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=ncol,
           frameon=False, handlelength=2.5, fontsize=14, mode='expand', borderaxespad=0.2)
  fig.tight_layout(rect=(0.0, 0.15, 1.0,1.0))
  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)


problems = []

with open("../cutest_problem_names.log", 'r') as f:
    for line in f.readlines():
      splitted = line.split(',')
      problems.append((splitted[0], int(splitted[1]), int(splitted[2])))

directories = os.listdir(data_dir)

print("[--] Num files: ", len(directories))

min_values = {}
init_values = {}
values = {}
for (function, d_idx, d) in problems:
  num_directions = [2, d//3, d//2, round((4/5) * d),  d]
  if f"{function}_{d_idx}_{d}" not in min_values:
    min_values[f"{function}_{d_idx}_{d}"] = None
  if f"{function}_{d_idx}_{d}" not in values:
    values[f"{function}_{d_idx}_{d}"] = {dir_type : [] for dir_type in direction_type}

  if f"{function}_{d_idx}_{d}" not in init_values:
    init_values[f"{function}_{d_idx}_{d}"] = None
  else:
    print("ALREADY EXISTS")
    exit()
  for dir_type in direction_type:
    if f"{function}_{d_idx}_{d}" not in min_values:
      break
    for (i, l) in enumerate(num_directions):
      dir_values = []
      min_v = None
      for r in range(reps):
        if os.path.exists(f"{data_dir}/{function}_{d_idx}_{d}_{dir_type}_{l}_1e-07_{r}.csv"):
          data = pd.read_csv(f"{data_dir}/{function}_{d_idx}_{d}_{dir_type}_{l}_1e-07_{r}.csv", header=None)
          val_iters = data[2].values
          feval_val = []
          for (val_idx, val) in enumerate(val_iters[:-1]):
            feval_val += [val for _ in range(data[3].values[val_idx + 1])]
          feval_val += [val_iters[-1]]

          dir_values.append(feval_val)

          min_val = data[1].values[0]
          f0 = data[0].values[0]
          if init_values[f"{function}_{d_idx}_{d}"] is None:
            print(f"[--] Loading results for {function}_{d_idx}_{d}")
            init_values[f"{function}_{d_idx}_{d}"] = f0
          if min_v is None or min_val < min_v:
            min_v = min_val
      if min_v is not None:
        min_len = min([len(dir_values[i]) for i in range(len(dir_values))])
        dir_values = [dir_values[i][:min_len] for i in range(len(dir_values))]
        mu_vals, std_vals = np.mean(dir_values, axis=0), np.std(dir_values, axis=0)
        min_values[f"{function}_{d_idx}_{d}"] = min(min_values[f"{function}_{d_idx}_{d}"], min_v) if min_values[f"{function}_{d_idx}_{d}"] is not None else min_v
        values[f"{function}_{d_idx}_{d}"][dir_type].append((mu_vals, std_vals))
      else:
        if f"{function}_{d_idx}_{d}" in init_values:
          del min_values[f"{function}_{d_idx}_{d}"]

          del init_values[f"{function}_{d_idx}_{d}"]

          del values[f"{function}_{d_idx}_{d}"]
        break


print("[--] Results Loaded!")
taus_perf = np.logspace(-15, 0, 1000)


results = {dir_type : [[None for _ in range(taus_perf.shape[0])] for _ in range(len(num_directions)) ] for dir_type in direction_type}
for (j, tau) in enumerate(taus_perf):
  for (k, fun_key) in enumerate(min_values.keys()):
    for i in range(len(num_directions)):
      min_val = min_values[fun_key]
      init_val = init_values[fun_key]
      for dir_type in direction_type:
        mu_values, _ = values[fun_key][dir_type][i]
        if np.any(np.isnan([np.min(mu_values), init_val,min_val])) or np.any(np.isinf([np.min(mu_values), init_val,min_val])):
          continue
        convergence_test = ((np.min(mu_values) - min_val + 1e-10) / (init_val - min_val + 1e-10)) <= tau
        if results[dir_type][i][j] is None:
          results[dir_type][i][j] = [convergence_test]
        else:
          results[dir_type][i][j].append(convergence_test)


taus = np.array([1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1e0])
full_values = {dir_type : [[None for _ in range(taus.shape[0])] for _ in range(len(num_directions)) ] for dir_type in direction_type}
for (j, tau) in enumerate(taus):
  for (k, fun_key) in enumerate(min_values.keys()):
    for i in range(len(num_directions)):
      min_val = min_values[fun_key]
      init_val = init_values[fun_key]
      for dir_type in direction_type:
        mu_values, _ = values[fun_key][dir_type][i]
        if np.any(np.isnan([np.min(mu_values), init_val,min_val])) or np.any(np.isinf([np.min(mu_values), init_val,min_val])):
          continue
        full_convergence_test = ((mu_values - min_val + 1e-10) / (init_val - min_val + 1e-10))  <= tau
        if full_values[dir_type][i][j] is None:
          full_values[dir_type][i][j] = [full_convergence_test]
        else:
          full_values[dir_type][i][j].append(full_convergence_test)


print("[--] Plotting Fraction of Solved")

fig, axes = plt.subplots(1, len(num_directions), figsize=(16, 4))
axes = axes.flatten()

for i in range(len(num_directions)):
  ax = axes[i]
  for dir_type in direction_type:
    ax.plot(taus_perf, [np.mean(ris) for ris in results[dir_type][i]], '-', lw=2.5,
            c=COLOR[dir_type], rasterized=True)
  ax.set_xscale('log')
  ax.set_ylim([-0.02, 1.02])
style_axes(axes, r"$\tau$", r"$\rho_{val}(\tau)$")
save_with_legend(fig, "./frac_solved_cutest.pdf")


print("[++] Plotting Fraction Solved per Function Evaluations")

for tau_idx in [0, 1, 2, 3, 4, 5]:
  print("\t[--] Accuracy tau = ", taus[tau_idx])
  fig, axes = plt.subplots(1, len(num_directions), figsize=(16, 4))
  axes = axes.flatten()

  for i in range(len(num_directions)):
    ax = axes[i]
    for dir_type in direction_type:
      max_len = max([elem.shape[0] for elem in full_values[dir_type][i][tau_idx]])
      for k in range(len(full_values[dir_type][i][tau_idx])):
        if full_values[dir_type][i][tau_idx][k].shape[0] < max_len:
          full_values[dir_type][i][tau_idx][k] = np.pad(full_values[dir_type][i][tau_idx][k], (0, max_len - full_values[dir_type][i][tau_idx][k].shape[0]), mode='maximum' )
      vals = np.mean(full_values[dir_type][i][tau_idx], axis=0)
      ax.plot(range(1, len(vals) + 1), vals, '-', lw=3, c=COLOR[dir_type], rasterized=True)
    ax.set_ylim([-0.01, 1.01])
  style_axes(axes, "# function evaluations", r"$\rho_{val}(\tau)$")
  save_with_legend(fig, f"./frac_solved_fvals_profile_cutest_{taus[tau_idx]}.pdf")


print("[++] Writing LaTeX table (l = d)")

# fraction of solved problems per method for l = d (last entry of num_directions), at the table taus
L_IDX = len(num_directions) - 1
frac = {}
for dir_type in direction_type:
  tests = [[] for _ in TABLE_TAUS]
  for fun_key in min_values.keys():
    min_val, init_val = min_values[fun_key], init_values[fun_key]
    mu_values, _ = values[fun_key][dir_type][L_IDX]
    best = np.min(mu_values)
    if not np.all(np.isfinite([best, init_val, min_val])):
      continue
    ratio = (best - min_val + 1e-10) / (init_val - min_val + 1e-10)
    for k, tau in enumerate(TABLE_TAUS):
      tests[k].append(ratio <= tau)
  frac[dir_type] = [np.mean(x) for x in tests]


def cell(methods, k):
  """mean +/- std (over the methods of the group) of the fraction of solved problems at TABLE_TAUS[k]."""
  v = np.array([frac[m][k] for m in methods])
  return rf"${v.mean():.2f} \pm {v.std():.2f}$"


def tau_header(tau):
  return rf"$\tau = 10^{{{int(round(np.log10(tau)))}}}$"


lines = [r"\begin{tabular}{l" + "c" * len(TABLE_TAUS) + "}",
         r"\toprule",
         " & " + " & ".join(tau_header(t) for t in TABLE_TAUS) + r" \\",
         r"\midrule"]
for name, methods in (("Structured", STRUCTURED), ("Unstructured", UNSTRUCTURED)):
  lines.append(name + " & " + " & ".join(cell(methods, k) for k in range(len(TABLE_TAUS))) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]

table = "\n".join(lines)
with open(table_path, "w") as f:
  f.write(table + "\n")
print(table)
print(f"[++] Table written to {table_path}")