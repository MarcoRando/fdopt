import os
import sys
import warnings

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

# usage: python plot_convergence_table.py <base_dir> <reps> [table.tex] [reps|boot]
#   reps : band / +- from the repetitions (one fraction of solved problems per repetition)
#   boot : band / +- from a bootstrap over the problems (uncertainty due to the choice of the benchmark)
base_dir = sys.argv[1]
reps = int(sys.argv[2])
table_path = sys.argv[3] if len(sys.argv) > 3 else "./frac_solved_table.tex"
BAND = sys.argv[4] if len(sys.argv) > 4 else "reps"
if BAND not in ("reps", "boot"):
  raise SystemExit("band must be 'reps' or 'boot'")
data_dir = f"{base_dir}/ffd/cutest"

EPS = 1e-10          # same regularisation as before in the convergence test
PCT = (5, 95)        # band = 5-95 percentiles
N_BOOT = 1000

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


def plot_band(ax, x, mean, lo, hi, color, lw):
  ax.plot(x, mean, '-', lw=lw, c=color, rasterized=True)
  ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0, rasterized=True)


problems = []

with open("../cutest_problem_names.log", 'r') as f:
    for line in f.readlines():
      splitted = line.split(',')
      problems.append((splitted[0], int(splitted[1]), int(splitted[2])))

directories = os.listdir(data_dir)

print("[--] Num files: ", len(directories))

# values[problem][method][l_idx] = array (n_reps_found, n_fevals): ALL the repetitions are kept
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
        dir_values = np.array([dir_values[i][:min_len] for i in range(len(dir_values))])    # (n_reps_found, min_len)
        min_values[f"{function}_{d_idx}_{d}"] = min(min_values[f"{function}_{d_idx}_{d}"], min_v) if min_values[f"{function}_{d_idx}_{d}"] is not None else min_v
        values[f"{function}_{d_idx}_{d}"][dir_type].append(dir_values)
      else:
        if f"{function}_{d_idx}_{d}" in init_values:
          del min_values[f"{function}_{d_idx}_{d}"]

          del init_values[f"{function}_{d_idx}_{d}"]

          del values[f"{function}_{d_idx}_{d}"]
        break


print("[--] Results Loaded!")

keys = list(min_values.keys())
P = len(keys)


def run_ratio(runs, key):
  """normalised optimality gap of every run: (f - f_min + eps) / (f_0 - f_min + eps)"""
  return (runs - min_values[key] + EPS) / (init_values[key] - min_values[key] + EPS)


_ratio_cache = {}


def best_ratios(dir_type, l_idx):
  """(P, reps) best normalised gap reached by each repetition on each problem (NaN = missing / invalid)."""
  if (dir_type, l_idx) not in _ratio_cache:
    out = np.full((P, reps), np.nan)
    for p, key in enumerate(keys):
      r = run_ratio(values[key][dir_type][l_idx], key).min(axis=1)
      r[~np.isfinite(r)] = np.nan
      out[p, :len(r)] = r
    _ratio_cache[(dir_type, l_idx)] = out
  return _ratio_cache[(dir_type, l_idx)]


def boot_means(s):
  """bootstrap over the problems: s is (P, T) = per-problem fraction solved -> (N_BOOT, T) resampled means."""
  rng = np.random.default_rng(0)
  n = s.shape[0]
  W = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)
  return (W @ s) / n


def summarize(S):
  """S: (P, reps, T), 1 = solved, 0 = not solved, NaN = missing. Returns mean, lo, hi, each of shape (T,)."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    s = np.nanmean(S, axis=1)                         # (P, T) fraction of repetitions that solved the problem
    keep = ~np.isnan(s).all(axis=1)
    s = s[keep]
    mean = s.mean(axis=0)
    if BAND == "reps":
      f = np.nanmean(S[keep], axis=0)                 # (reps, T) one fraction of solved problems per repetition
      f = f[~np.isnan(f).all(axis=1)]
      lo, hi = np.percentile(f, PCT, axis=0)
    else:
      lo, hi = np.percentile(boot_means(s), PCT, axis=0)
  return mean, lo, hi


print(f"[--] Plotting Fraction of Solved (band: {BAND})")
taus_perf = np.logspace(-15, 0, 1000)

profile = {}
for dir_type in direction_type:
  for l_idx in range(len(num_dirs_label)):
    R = best_ratios(dir_type, l_idx)
    S = (R[:, :, None] <= taus_perf[None, None, :]).astype(np.float32)
    S[np.isnan(R)] = np.nan
    profile[(dir_type, l_idx)] = summarize(S)

fig, axes = plt.subplots(1, len(num_dirs_label), figsize=(16, 4))
axes = axes.flatten()

for i in range(len(num_dirs_label)):
  ax = axes[i]
  for dir_type in direction_type:
    mean, lo, hi = profile[(dir_type, i)]
    plot_band(ax, taus_perf, mean, lo, hi, COLOR[dir_type], 2.5)
  ax.set_xscale('log')
  ax.set_ylim([-0.02, 1.02])
style_axes(axes, r"$\tau$", r"$\rho_{val}(\tau)$")
save_with_legend(fig, "./frac_solved_cutest.pdf")


print("[++] Plotting Fraction Solved per Function Evaluations")
taus = np.array([1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1e0])
PLOT_TAU_IDX = [0, 1, 2, 3, 4, 5]

traj = {}
for dir_type in direction_type:
  for l_idx in range(len(num_dirs_label)):
    runs_list = [values[key][dir_type][l_idx] for key in keys]
    max_len = max(r.shape[1] for r in runs_list)
    R = np.full((P, reps, max_len), np.nan, dtype=np.float32)
    for p, key in enumerate(keys):
      rr = run_ratio(runs_list[p], key)
      n_r, L = rr.shape
      bad = ~np.isfinite(rr).all(axis=1)
      R[p, :n_r, :L] = rr
      R[p, :n_r, L:] = rr.min(axis=1)[:, None]        # a finished run keeps its best value (solved stays solved)
      R[p, :n_r][bad] = np.nan
    missing = np.isnan(R)
    for tau_idx in PLOT_TAU_IDX:
      S = (R <= taus[tau_idx]).astype(np.float32)
      S[missing] = np.nan
      traj[(dir_type, l_idx, tau_idx)] = summarize(S)
    del R, S, missing

for tau_idx in PLOT_TAU_IDX:
  print("\t[--] Accuracy tau = ", taus[tau_idx])
  fig, axes = plt.subplots(1, len(num_dirs_label), figsize=(16, 4))
  axes = axes.flatten()

  for i in range(len(num_dirs_label)):
    ax = axes[i]
    for dir_type in direction_type:
      mean, lo, hi = traj[(dir_type, i, tau_idx)]
      plot_band(ax, np.arange(1, len(mean) + 1), mean, lo, hi, COLOR[dir_type], 3)
    ax.set_ylim([-0.01, 1.01])
  style_axes(axes, "# function evaluations", r"$\rho_{val}(\tau)$")
  save_with_legend(fig, f"./frac_solved_fvals_profile_cutest_{taus[tau_idx]}.pdf")


print("[++] Writing LaTeX table")


def table_cell(l_idx, methods, tau):
  """mean +- std of the fraction of solved problems, averaged over the methods of the group.
  reps: std over the repetitions.  boot: std of the bootstrap over the problems (standard error)."""
  per_rep, per_prob = [], []
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for m in methods:
      R = best_ratios(m, l_idx)
      S = (R <= tau).astype(float)
      S[np.isnan(R)] = np.nan
      per_prob.append(np.nanmean(S, axis=1))          # (P,)
      per_rep.append(np.nanmean(S, axis=0))           # (reps,)
  if BAND == "reps":
    g = np.mean(per_rep, axis=0)
    g = g[~np.isnan(g)]
    mean, std = g.mean(), (g.std(ddof=1) if len(g) > 1 else 0.0)
  else:
    sg = np.mean(per_prob, axis=0)
    sg = sg[~np.isnan(sg)]
    mean, std = sg.mean(), boot_means(sg[:, None])[:, 0].std()
  return rf"${mean:.2f} \pm {std:.2f}$"


def tau_header(tau):
  return rf"$\tau = 10^{{{int(round(np.log10(tau)))}}}$"


lines = [r"\begin{tabular}{" + "l" * (len(TABLE_TAUS) + 1) + "}"]
for l_idx, l_lab in enumerate(num_dirs_label):
  lines += [r"\toprule",
            rf"$\ell = {l_lab}$ & " + " & ".join(tau_header(t) for t in TABLE_TAUS) + r" \\",
            r"\midrule"]
  for name, methods in (("Structured", STRUCTURED), ("Unstructured", UNSTRUCTURED)):
    lines.append(name + " & " + " & ".join(table_cell(l_idx, methods, t) for t in TABLE_TAUS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]

table = "\n".join(lines)
with open(table_path, "w") as f:
  f.write(table + "\n")
print(table)
print(f"[++] Table written to {table_path}")