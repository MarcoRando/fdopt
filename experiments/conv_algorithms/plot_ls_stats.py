"""Box plots for the results of run_conv_exp.py (FFD + Armijo line search).

Same layout as plot_grad_accuracy.py: one row per test function, one column per number of directions l, one box per
direction family (colours / letters / labels from plot_common).  Two figures by default:
    * number of line-search failures,
    * median step size.

Two data sources:
    * summary CSV (default, --csv): the statistics across repetitions are stored per iteration k.  For every
      (function, family, l) the box is taken at the last iteration still reached by at least --min-frac of the runs
      (runs stop after T function evaluations, so they do not all have the same number of iterations).
      The step size is therefore the step of that iteration.
    * raw CSV (--raw, written by run_conv_exp.py --raw-out): one value per repetition is computed first (final number
      of failures; median of the step size over the iterations of the run), and the box shows the distribution of
      these values across repetitions.  This is the more meaningful version of "median step size achieved".
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from matplotlib.patches import Patch

from plot_common import COLOR, LABEL, LETTER, NAN, ORDER, common_selection, l_label, plt, read_rows

# metric -> (column prefix, y label, file stem, per-run aggregation used with --raw)
METRICS = {
    "fails": ("fails", "number of failures", "failures", "last"),
    "step": ("step", "median step size", "stepsize", "median"),
    "gamma": ("gamma", r"line-search step $\gamma$", "gamma", "median"),
    "prog": ("prog", r"$\frac{F(x_k)-F^*}{F(x_0)-F^*}$", "progress", "last"),
}
PCT = ["p05", "p25", "p75", "p95", "min", "max"]


def pct_value(name):
    return 0.0 if name == "min" else 100.0 if name == "max" else float(name[1:])


def box_stats(r, prefix, box, band, floor):
    get = lambda s: r.get(f"{prefix}_{s}", NAN)
    vals = [get("median"), get(box[0]), get(box[1]), get(band[0]), get(band[1])]
    if any(not isinstance(v, float) or v != v for v in vals):
        return None
    med, q1, q3, lo, hi = vals
    return {"med": max(med, floor), "q1": max(q1, floor), "q3": max(q3, floor),
            "whislo": max(lo, floor), "whishi": max(hi, floor), "fliers": []}


def summary_stats(rows, prefix, a, floor):
    """(function, family, l) -> box statistics at the last iteration reached by >= min_frac of the runs."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["function"], r["family"], r["l"])].append(r)
    out = {}
    for key, rs in groups.items():
        nmax = max(r["n_reps"] for r in rs)
        ok = [r for r in rs if r["n_reps"] >= a.min_frac * nmax]
        st = box_stats(max(ok, key=lambda r: r["k"]), prefix, a.box, a.band, floor)
        if st is not None:
            out[key] = st
    return out


def raw_stats(rows, prefix, agg, a, floor):
    """(function, family, l) -> box statistics of one value per repetition (needs the raw CSV)."""
    runs = defaultdict(list)
    for r in rows:
        runs[(r["function"], r["family"], r["l"], r["rep"])].append((r["k"], r.get(prefix, NAN)))
    per_group = defaultdict(list)
    for (fn, fam, l, _), kv in runs.items():
        arr = np.array([v for _, v in sorted(kv)], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            per_group[(fn, fam, l)].append(arr[-1] if agg == "last" else float(np.median(arr)))
    out = {}
    for key, v in per_group.items():
        q = {p: float(np.percentile(v, pct_value(p))) for p in {*a.box, *a.band}}
        out[key] = {"med": max(float(np.median(v)), floor), "q1": max(q[a.box[0]], floor),
                    "q3": max(q[a.box[1]], floor), "whislo": max(q[a.band[0]], floor),
                    "whishi": max(q[a.band[1]], floor), "fliers": []}
    return out


def make_figure(stats, metric, functions, fams, ls, d, dtype, a):
    _, ylabel, stem, _ = METRICS[metric]
    log = metric in a.log
    nr, nc = len(functions), len(ls)
    left, w, gap, right = 1.5, 2.35, 0.72, 0.2          # left margin = tick labels + y label
    ph, xt, top = 2.0, 0.35, 0.1                        # plot height, room for the x tick labels, top margin
    ta = lambda i: 0.45 + (0.35 if i == 0 else 0.0)     # room above the plots: function name (+ l titles in row 0)
    W = left + nc * w + (nc - 1) * gap + right
    ncol_leg = max(2, min(len(fams), int(W // 2.6)))
    leg_h = 0.1 + 0.34 * ((len(fams) + ncol_leg - 1) // ncol_leg)
    H = top + sum(ta(i) + ph + xt for i in range(nr)) + leg_h
    fig = plt.figure(figsize=tuple(a.figsize) if a.figsize else (W, H))
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
            st, colors, pos = [], [], []
            for k, fam in enumerate(fams, start=1):
                s = stats.get((fn, fam, l))
                if s is not None:
                    st.append(s), colors.append(COLOR[fam]), pos.append(k)
            if st:
                bp = ax.bxp(st, positions=pos, widths=0.65, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=1.2),
                            capprops=dict(lw=1.2), boxprops=dict(lw=1.0))
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c), patch.set_alpha(0.9)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            ax.set_xlim(0.4, len(fams) + 0.6)
            ax.set_xticks(range(1, len(fams) + 1))
            ax.set_xticklabels([LETTER[f] for f in fams])
            if log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(l_label(l, d))
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
            elif a.sharey:
                ax.tick_params(labelleft=False)
        y = y_plot + ph + xt
    handles = [Patch(facecolor=COLOR[f], label=f"{LETTER[f]} = {LABEL[f]}") for f in fams]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol_leg, frameon=False,
               handlelength=1.5, fontsize=16)
    os.makedirs(a.outdir, exist_ok=True)
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{a.prefix}conv_{stem}_d{d}_{dtype}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {a.prefix}conv_{stem}_d{d}_{dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/convergence_ls.csv", help="summary CSV written by run_conv_exp.py")
    ap.add_argument("--raw", default=None,
                    help="raw CSV (run_conv_exp.py --raw-out): boxes then show the distribution of one value per run")
    ap.add_argument("--outdir", default=None, help="default: <csv dir>/figures")
    ap.add_argument("--functions", nargs="+", default=None, help="function names (rows); default: all in the CSV")
    ap.add_argument("--families", nargs="+", default=None, help="families to show; default: all in the CSV")
    ap.add_argument("--metrics", nargs="+", default=["fails", "step"], choices=list(METRICS))
    ap.add_argument("--d", type=int, default=None, help="dimension (default: the smallest in the CSV)")
    ap.add_argument("--dtype", default=None, help="float32 / float64 (default: the first in the CSV)")
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l to show (default: all)")
    ap.add_argument("--box", nargs=2, default=["p25", "p75"], choices=PCT, help="box limits")
    ap.add_argument("--band", nargs=2, default=["p05", "p95"], choices=PCT, help="whisker limits")
    ap.add_argument("--log", nargs="*", default=["step", "gamma", "prog"], choices=list(METRICS),
                    help="metrics on a log y axis (the others are linear)")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="summary CSV: use the last iteration reached by at least this fraction of the runs")
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    ap.add_argument("--figsize", type=float, nargs=2, default=None, help="override the automatic figure size")
    ap.add_argument("--floor", type=float, default=1e-16, help="values are clipped below this (log axes)")
    ap.add_argument("--prefix", default="", help="prefix for the output file names")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args()
    src = a.raw or a.csv
    a.outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(src)), "figures")

    rows = read_rows(src)
    if not rows:
        raise SystemExit(f"no rows in {src}")
    functions, d, dtype, fams, ls, sub = common_selection(rows, a.functions, a.d, a.dtype, a.families, a.ls)
    for metric in a.metrics:
        prefix, _, _, agg = METRICS[metric]
        floor = a.floor if metric in a.log else -np.inf
        if a.raw:
            if prefix not in sub[0]:
                print(f"[skip] metric '{metric}': column '{prefix}' not in {src}")
                continue
            stats = raw_stats(sub, prefix, agg, a, floor)
        else:
            if f"{prefix}_median" not in sub[0]:
                print(f"[skip] metric '{metric}': columns {prefix}_* not in {src}")
                continue
            stats = summary_stats(sub, prefix, a, floor)
        make_figure(stats, metric, functions, fams, ls, d, dtype, a)


if __name__ == "__main__":
    main()