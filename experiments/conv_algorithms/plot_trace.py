import argparse
import glob
import os

import numpy as np
from matplotlib.lines import Line2D

from plot_common import COLOR, LABEL, ORDER, common_selection, l_label, plt, read_rows

try:                                                    # same style and font sizes as plot_grad_accuracy.py
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 9, "legend.fontsize": 12, "grid.alpha": 0.5})

YLABEL = r"$\frac{F(x_k)-F^*}{F(x_0)-F^*}$"
METRICS = {   # key -> (column prefix, y label, file stem)
    "prog": ("prog", YLABEL, "conv_curves"),
    "step": ("gamma", "step size", "step_curves"),
}
XTICKS = [0, 5, 10, 15, 20]                           # fixed x ticks, in units of evaluations / (d + 1)
PCT = ["p05", "p25", "p75", "p95", "min", "max"]


def load_rows(a):
    paths = sorted(glob.glob(os.path.join(a.trace_dir, "trace_*.csv"))) if a.trace_dir else [a.csv]
    if not paths:
        raise SystemExit(f"no trace_*.csv files in {a.trace_dir}")
    rows = []
    for p in paths:
        rows += read_rows(p)
    return rows, paths


def curve(sub, fn, fam, l, d, prefix, a):
    rs = [r for r in sub if r["function"] == fn and r["family"] == fam and r["l"] == l]
    if not rs:
        return None
    nmax = max(r["n_reps"] for r in rs)
    rs = sorted((r for r in rs if r["n_reps"] >= a.min_frac * nmax), key=lambda r: r["k"])
    x = np.array([r["evals_median"] for r in rs], dtype=float) / (d + 1)   # evaluations per (d+1)
    y = [np.array([r[f"{prefix}_{s}"] for r in rs], dtype=float) for s in ("median", a.band[0], a.band[1])]
    ok = np.isfinite(x) & np.isfinite(y[0])
    if not ok.any():
        return None
    return (x[ok], *[np.maximum(v[ok], a.floor) for v in y])


def make_figure(sub, metric, functions, fams, ls, d, dtype, a):
    prefix, ylabel, stem0 = METRICS[metric]
    nr, nc = len(functions), len(ls)
    left, w, gap, right = 1.5, 2.35, 0.72, 0.2          # left margin = tick labels + the fraction used as y label
    ph, xt, top = 2.0, 0.35, 0.1                        # plot height, room for the x tick labels, top margin
    ta = lambda i: 0.45 + (0.35 if i == 0 else 0.0)     # room above the plots: function name (+ l titles in row 0)
    W = left + nc * w + (nc - 1) * gap + right
    ncol_leg = max(2, min(len(fams), int(W // 2.6)))
    leg_h = 0.1 + 0.34 * ((len(fams) + ncol_leg - 1) // ncol_leg)
    H = top + sum(ta(i) + ph + xt for i in range(nr)) + leg_h
    fig = plt.figure(figsize=tuple(a.figsize))
    ds = "default" if a.no_steps else "steps-post"
    y = top
    for i, fn in enumerate(functions):
        fig.text((left + (nc * w + (nc - 1) * gap) / 2) / W, 1 - (y + 0.05) / H, fn, ha="center", va="top",
                 fontsize=16, fontweight="bold")
        y_plot = y + ta(i)
        first = None
        for j, l in enumerate(ls):
            ax = fig.add_axes([(left + j * (w + gap)) / W, 1 - (y_plot + ph) / H, w / W, ph / H],
                              sharey=first if (a.sharey and j > 0) else None)
            first = ax if j == 0 else first
            any_curve = False
            for fam in fams:
                c = curve(sub, fn, fam, l, d, prefix, a)
                if c is None:
                    continue
                any_curve = True
                x, med, lo, hi = c
                ax.fill_between(x, lo, hi, color=COLOR[fam], alpha=a.alpha, lw=0,
                                step=None if a.no_steps else "post")
                ax.plot(x, med, color=COLOR[fam], lw=3, drawstyle=ds)
            if not any_curve:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            if metric in a.linear:
                print("")
#                ax.set_ylim(bottom=0)
            else:
                ax.set_yscale("log")
            ax.set_xlim(0, XTICKS[-1])
            ax.set_xticks(XTICKS)
            if i == 0:
                ax.set_title(l_label(l, d))
            if i == nr - 1:
                ax.set_xlabel(r"$\#\ \mathrm{evals} / (d + 1)$", fontsize=14)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=16, labelpad=8)
            elif a.sharey:
                ax.tick_params(labelleft=False)
        y = y_plot + ph + xt
    handles = [Line2D([0], [0], color=COLOR[f], lw=3, label=LABEL[f]) for f in fams]
    # fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.0), ncol=ncol_leg, frameon=False,
    #            handlelength=1.5, fontsize=16)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=5,
            frameon=False, handlelength=2.5, fontsize=14, mode='expand', borderaxespad=0.2)

    fig.tight_layout(rect=(0.0, 0.15, 1.0,1.0))
    os.makedirs(a.outdir, exist_ok=True)
    tag = "-".join(functions) if a.functions else "all"
    stem = f"{a.prefix}{stem0}_{tag}_d{d}_{dtype}"
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {stem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/convergence_ls.csv", help="summary CSV written by run_conv_exp.py")
    ap.add_argument("--trace-dir", default=None,
                    help="directory with the trace_*.csv files (run_conv_exp.py --trace-dir); used instead of --csv")
    ap.add_argument("--outdir", default=None, help="default: <csv or trace dir>/figures")
    ap.add_argument("--functions", nargs="+", default=None, help="targets to plot (rows); default: all in the data")
    ap.add_argument("--families", nargs="+", default=None, help="families to show; default: all in the data")
    ap.add_argument("--metrics", nargs="+", default=["prog", "step"], choices=list(METRICS),
                    help="prog: normalised function value; step: step size")
    ap.add_argument("--linear", nargs="*", default=["step"], choices=list(METRICS), help="metrics on a linear y axis (the others are log)")
    ap.add_argument("--d", type=int, default=None, help="dimension (default: the smallest in the data)")
    ap.add_argument("--dtype", default=None, help="float32 / float64 (default: the first in the data)")
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l to show (default: all)")
    ap.add_argument("--band", nargs=2, default=["p25", "p75"], choices=PCT, help="shaded band limits")
    ap.add_argument("--alpha", type=float, default=0.2, help="opacity of the band")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="drop the final steps reached by fewer than this fraction of the runs")
    ap.add_argument("--no-steps", action="store_true", help="join the points with straight lines (default: steps)")
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    ap.add_argument("--figsize", type=float, nargs=2, default=[14, 10], help="figure size in inches (as plot_grad_accuracy)")
    ap.add_argument("--floor", type=float, default=1e-16, help="values are clipped below this (log axis)")
    ap.add_argument("--prefix", default="", help="prefix for the output file names")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args()
    a.outdir = a.outdir or os.path.join(os.path.abspath(a.trace_dir or os.path.dirname(os.path.abspath(a.csv))),
                                        "figures")

    rows, paths = load_rows(a)
    if not rows:
        raise SystemExit(f"no rows in {paths}")
    functions, d, dtype, fams, ls, sub = common_selection(rows, a.functions, a.d, a.dtype, a.families, a.ls)
    for metric in a.metrics:
        prefix = METRICS[metric][0]
        need = ["evals_median", f"{prefix}_median", f"{prefix}_{a.band[0]}", f"{prefix}_{a.band[1]}"]
        miss = [c for c in need if c not in sub[0]]
        if miss:
            print(f"[skip] metric '{metric}': columns {miss} not found "
                  f"(trace files need the '{prefix}' statistics; available: {sorted(sub[0])})")
            continue
        make_figure(sub, metric, functions, fams, ls, d, dtype, a)

if __name__ == "__main__":
    main()