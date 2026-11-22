import argparse
import glob
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_common import COLOR, LABEL, ORDER, common_selection, l_label, plt, read_rows

try:                                                    # same style and font sizes as plot_grad_accuracy.py
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "grid.alpha": 0.5})

YLABEL = r"$\frac{F(x_k)-F^*}{F(x_0)-F^*}$"
XLABEL = r"$\#\ \mathrm{evals} / (d + 1)$"
METRICS = {   # key -> (column prefix, y label, file stem)
    "prog": ("prog", YLABEL, "conv_curves"),                              # curves: normalised function value
    "fails": ("fails", "cumulative\n# of failures", "fails_curves"),      # curves: cumulative number of failures
    "step": ("step", "step size", "step_boxplot"),                        # boxplots at STEP_TICKS + LaTeX tables
}
XTICKS = [0, 5, 10, 15, 20]                             # fixed x ticks of the curves, in evaluations / (d + 1)
STEP_TICKS = [5, 10, 15, 20]                            # evaluations / (d + 1) at which the step size is shown
PCT = ["p05", "p25", "p75", "p95", "min", "max"]


fn_to_title = {
    'LeastSquares' : "Least Squares",
    'Qing' : "Qing",
    'Rosenbrock' : 'Rosenbrock',
    'RegularizedLogistic' : 'Logistic Regression',
    'Trid' : 'Trid',
    'Griewank' : 'Griewank'
}


def load_rows(a):
    paths = sorted(glob.glob(os.path.join(a.trace_dir, "trace_*.csv"))) if a.trace_dir else [a.csv]
    if not paths:
        raise SystemExit(f"no trace_*.csv files in {a.trace_dir}")
    rows = []
    for p in paths:
        rows += read_rows(p)
    return rows, paths


def rows_of(sub, fn, fam, l, a):

    rs = [r for r in sub if r["function"] == fn and r["family"] == fam and r["l"] == l]
    if not rs:
        return []
    nmax = max(r["n_reps"] for r in rs)
    return sorted((r for r in rs if r["n_reps"] >= a.min_frac * nmax), key=lambda r: r["k"])


def curve(rs, d, prefix, a):
    x = np.array([r["evals_median"] for r in rs], dtype=float) / (d + 1)   # evaluations per (d+1)
    y = [np.array([r[f"{prefix}_{s}"] for r in rs], dtype=float) for s in ("median", a.band[0], a.band[1])]
    ok = np.isfinite(x) & np.isfinite(y[0])
    if not ok.any():
        return None
    return (x[ok], *[np.maximum(v[ok], a.floor) for v in y])


def step_at(rs, target):

    if not rs:
        return None
    ev = np.array([r["evals_median"] for r in rs], dtype=float)
    if not np.isfinite(ev).any() or target > 1.02 * np.nanmax(ev):
        return None
    return rs[int(np.nanargmin(np.abs(ev - target)))]


def box_stats(r, prefix, box, band):
    get = lambda s: r.get(f"{prefix}_{s}", float("nan"))
    vals = [get("median"), get(box[0]), get(box[1]), get(band[0]), get(band[1])]
    if any(not isinstance(v, float) or v != v for v in vals):
        return None
    med, q1, q3, lo, hi = vals
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": []}


def grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0):

    nr, nc = len(functions), len(ls)
    left, w, gap, right = 1.5, 2.35, 0.72, 0.2          # left margin = tick labels + the fraction used as y label
    ph, xt, top = 2.0, 0.35, 0.1                        # plot height, room for the x tick labels, top margin
    ta = lambda i: 0.45 + (0.35 if i == 0 else 0.0)     # room above the plots: function name (+ l titles in row 0)
    W = left + nc * w + (nc - 1) * gap + right
    leg_h = 0.1 + 0.34 * 2
    H = top + sum(ta(i) + ph + xt for i in range(nr)) + leg_h
    fig = plt.figure(figsize=tuple(a.figsize))
    y = top
    for i, fn in enumerate(functions):
        fig.text((left + (nc * w + (nc - 1) * gap) / 2) / W, 1 - (y + 0.05) / H, fn_to_title[fn], ha="center", va="top",
                 fontsize=16, fontweight="bold")
        y_plot = y + ta(i)
        first = None
        for j, l in enumerate(ls):
            ax = fig.add_axes([(left + j * (w + gap)) / W, 1 - (y_plot + ph) / H, w / W, ph / H],
                              sharey=first if (a.sharey and j > 0) else None)
            first = ax if j == 0 else first
            if not draw(ax, fn, l):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            if i == 0:
                ax.set_title(l_label(l, d))
            if i == nr - 1:
                ax.set_xlabel(XLABEL, fontsize=14)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
            elif a.sharey:
                ax.tick_params(labelleft=False)
        y = y_plot + ph + xt
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=4,
               frameon=False, handlelength=2.5, fontsize=16, mode='expand', borderaxespad=0.3)
#    fig.tight_layout(rect=(0.0, 0.5, 1.0, 1.0))
    os.makedirs(a.outdir, exist_ok=True)
    tag = "-".join(functions) if a.functions else "all"
    stem = f"{a.prefix}{stem0}_{tag}_d{d}_{dtype}"
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {stem}")


def make_curves(sub, metric, functions, fams, ls, d, dtype, a):
    prefix, ylabel, stem0 = METRICS[metric]
    log = metric not in a.linear
    ds = "default" if a.no_steps else "steps-post"

    def draw(ax, fn, l):
        any_curve = False
        for fam in fams:
            c = curve(rows_of(sub, fn, fam, l, a), d, prefix, a)
            if c is None:
                continue
            any_curve = True
            x, med, lo, hi = c
            ax.plot(x, med, color=COLOR[fam], lw=3, drawstyle=ds, rasterized=True)
            ax.fill_between(x, lo, hi, color=COLOR[fam], alpha=a.alpha, lw=0, step=None if a.no_steps else "post", rasterized=True)
        ax.set_xlim(0, XTICKS[-1])
        ax.set_xticks(XTICKS)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        return any_curve

    handles = [Line2D([0], [0], color=COLOR[f], lw=3, label=LABEL[f]) for f in fams]
    grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0)


def make_step_boxplots(sub, functions, fams, ls, d, dtype, a):

    _, ylabel, stem0 = METRICS["step"]
    n = len(fams)
    spacing = STEP_TICKS[1] - STEP_TICKS[0]
    wb = 0.8 * spacing / n

    def draw(ax, fn, l):
        stats, colors, pos = [], [], []
        for t in STEP_TICKS:
            for k, fam in enumerate(fams):
                r = step_at(rows_of(sub, fn, fam, l, a), t * (d + 1))
                st = box_stats(r, "step", a.box, a.band) if r is not None else None
                if st is not None:
                    stats.append(st), colors.append(COLOR[fam]), pos.append(t + (k - (n - 1) / 2) * wb)
        if stats:
            bp = ax.bxp(stats, positions=pos, widths=0.85 * wb, patch_artist=True, showfliers=False,
                        manage_ticks=False, medianprops=dict(color="black", lw=1.4), whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2), boxprops=dict(lw=0.8))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c), patch.set_alpha(0.9)
        ax.set_xlim(STEP_TICKS[0] - 0.6 * spacing, STEP_TICKS[-1] + 0.6 * spacing)
        ax.set_xticks(STEP_TICKS)
        ax.set_ylim(bottom=0)
        return bool(stats)

    handles = [Patch(facecolor=COLOR[f], label=LABEL[f]) for f in fams]
    grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0)


def fmt_num(v):
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    e = int(e)
    return f"{v:.3g}" if -2 <= e <= 2 else rf"${m}\cdot 10^{{{e}}}$"


def tex(s):
    return str(s).replace("_", r"\_")


def write_step_tables(sub, functions, fams, ls, d, dtype, a):

    out = ["% median step size; needs \\usepackage{booktabs,graphicx}", ""]
    nt = len(STEP_TICKS)
    ticks = ", ".join(str(t) for t in STEP_TICKS)
    for fn in functions:
        head1 = "Method & " + " & ".join(f"\\multicolumn{{{nt}}}{{c}}{{{l_label(l, d)}}}" for l in ls) + " \\\\"
        head2 = " & " + " & ".join(" & ".join(str(t) for t in STEP_TICKS) for _ in ls) + " \\\\"
        body = []
        for fam in fams:
            cells = []
            for l in ls:
                rs = rows_of(sub, fn, fam, l, a)
                for t in STEP_TICKS:
                    r = step_at(rs, t * (d + 1))
                    cells.append(fmt_num(r["step_median"]) if r is not None else "--")
            body.append(f"{tex(LABEL[fam])} & " + " & ".join(cells) + " \\\\")
        spec = "l|" + "|".join("c" * nt for _ in ls)
        cap = (f"{tex(fn)}: median step size ($d={d}$, {dtype}) after $\\#\\,\\mathrm{{evals}}/(d+1) \\in \\{{{ticks}\\}}$ "
               f"function evaluations, for different numbers of directions $\\ell$. `--' = the runs stop before.")
        out += ["\\begin{table}[ht]", "\\centering", "\\footnotesize", "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{%s}" % spec, "\\toprule", head1, head2, "\\midrule"] + body + \
               ["\\bottomrule", "\\end{tabular}}", "\\caption{%s}" % cap,
                "\\label{tab:conv-step-%s}" % fn.lower(), "\\end{table}", ""]
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{a.prefix}step_tables_d{d}_{dtype}.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"[ok] {os.path.basename(path)} -> {a.outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/convergence_ls.csv", help="summary CSV written by run_conv_exp.py")
    ap.add_argument("--trace-dir", default=None,
                    help="directory with the trace_*.csv files (run_conv_exp.py --trace-dir); used instead of --csv")
    ap.add_argument("--outdir", default=None, help="default: <csv or trace dir>/figures")
    ap.add_argument("--functions", nargs="+", default=None, help="targets to plot (rows); default: all in the data")
    ap.add_argument("--families", nargs="+", default=None, help="families to show; default: all in the data")
    ap.add_argument("--metrics", nargs="+", default=["prog", "fails", "step"], choices=list(METRICS),
                    help="prog: normalised function value; fails: cumulative failures (curves); "
                         "step: step size (boxplots at 5, 10, 15, 20 evals/(d+1) + LaTeX tables)")
    ap.add_argument("--linear", nargs="*", default=["fails"], choices=["prog", "fails"],
                    help="curve metrics on a linear y axis (the others are log)")
    ap.add_argument("--d", type=int, default=None, help="dimension (default: the smallest in the data)")
    ap.add_argument("--dtype", default=None, help="float32 / float64 (default: the first in the data)")
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l to show (default: all)")
    ap.add_argument("--band", nargs=2, default=["p05", "p95"], choices=PCT,
                    help="shaded band of the curves / error bars (whiskers) of the boxplots")
    ap.add_argument("--box", nargs=2, default=["p25", "p75"], choices=PCT, help="box limits of the step boxplots")
    ap.add_argument("--alpha", type=float, default=0.2, help="opacity of the band")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="drop the final steps reached by fewer than this fraction of the runs")
    ap.add_argument("--no-steps", action="store_true", help="join the points with straight lines (default: steps)")
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    ap.add_argument("--no-tables", action="store_true", help="do not write the LaTeX tables of the step size")
    ap.add_argument("--figsize", type=float, nargs=2, default=[15, 11], help="figure size in inches (as plot_grad_accuracy)")
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
        if metric == "step":
            need += [f"{prefix}_{a.box[0]}", f"{prefix}_{a.box[1]}"]
        miss = [c for c in need if c not in sub[0]]
        if miss:
            print(f"[skip] metric '{metric}': columns {miss} not found "
                  f"(trace files need the '{prefix}' statistics; available: {sorted(sub[0])})")
            continue
        if metric == "step":
            make_step_boxplots(sub, functions, fams, ls, d, dtype, a)
            if not a.no_tables:
                write_step_tables(sub, functions, fams, ls, d, dtype, a)
        else:
            make_curves(sub, metric, functions, fams, ls, d, dtype, a)


if __name__ == "__main__":
    main()