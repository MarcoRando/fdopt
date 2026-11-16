import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

NAN = float("nan")
from plot_common import COLOR, LABEL, LETTER, ORDER  # noqa: E402  (same colours as every other figure)

METRICS = {  
    "err": ("err", r"$\frac{\|g(x,P,h)-\nabla F(x)\|}{\|\nabla F(x)\|}$", "grad_error"),
    "cover": ("cover", r"$\frac{\|\nabla F(x)- \frac{1}{\alpha}\,PP^\top\nabla F(x)\|}{\|\nabla F(x)\|}$", "coverage_error"),
}
PCT = ["p05", "p25", "p75", "p95", "min", "max"]

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 9, "legend.fontsize": 12, "grid.alpha": 0.5})


def read_rows(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in ("function", "family", "dtype"):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = NAN
            r["d"], r["l"] = int(r["d"]), int(r["l"])
            rows.append(r)
    return rows


def l_label(l, d):
    if l == 2:
        return r"$\ell = 2$"
    for num, den, txt in ((1, 1, "d"), (4, 5, "4d/5"), (1, 2, "d/2"), (1, 3, "d/3")):
        if l == (num * d) // den:
            return rf"$\ell = {txt}$" if txt != "4d/5" else r"$\ell = 4d/5$"
    return rf"$\ell = {l}$"


def box_stats(r, prefix, box, band, floor):
    get = lambda s: r.get(f"{prefix}_{s}", NAN)
    vals = [get("median"), get(box[0]), get(box[1]), get(band[0]), get(band[1])]
    if any(v != v for v in vals):
        return None
    med, q1, q3, lo, hi = vals
    return {"med": max(med, floor), "q1": max(q1, floor), "q3": max(q3, floor),
            "whislo": max(lo, floor), "whishi": max(hi, floor), "fliers": []}


def make_figure(rows, metric, functions, fams, ls, d, dtype, a, linear=False):

    prefix, ylabel, stem = METRICS[metric]
    nr, nc = len(functions), len(ls)
    left, w, gap, right = 1.5, 2.35, 0.72, 0.2          # left margin = tick labels + the fraction used as y label
    ph, xt, top = 2.0, 0.35, 0.1                        # plot height, room for the x tick labels, top margin
    ta = lambda i: 0.45 + (0.35 if i == 0 else 0.0)     # room above the plots: function name (+ l titles in row 0)
    W = left + nc * w + (nc - 1) * gap + right
    ncol_leg =  max(2, min(len(fams), int(W // 2.6)))
    leg_h = 0.1 + 0.34 * ((len(fams) + ncol_leg - 1) // ncol_leg)
    H = top + sum(ta(i) + ph + xt for i in range(nr)) + leg_h
    fig = plt.figure(figsize= (15, 10) )#(W, H))
    y = top
    for i, fn in enumerate(functions):
        fig.text((left + (nc * w + (nc - 1) * gap) / 2) / W, 1 - (y + 0.05) / H, fn, ha="center", va="top",
                 fontsize=16, fontweight="bold")
        y_plot = y + ta(i)
        for j, l in enumerate(ls):
            ax = fig.add_axes([(left + j * (w + gap)) / W, 1 - (y_plot + ph) / H, w / W, ph / H])
            sel = {r["family"]: r for r in rows
                   if r["function"] == fn and r["d"] == d and r["dtype"] == dtype and r["l"] == l}
            stats, colors, pos = [], [], []
            for k, fam in enumerate(fams, start=1):
                st = box_stats(sel[fam], prefix, a.box, a.band, a.floor) if fam in sel else None
                if st is not None:
                    stats.append(st), colors.append(COLOR[fam]), pos.append(k)
            if stats:
                bp = ax.bxp(stats, positions=pos, widths=0.65, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=1.2),
                            capprops=dict(lw=1.2), boxprops=dict(lw=1.0))
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c), patch.set_alpha(0.9)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            ax.set_xlim(0.4, len(fams) + 0.6)
            ax.set_xticks(range(1, len(fams) + 1))
            ax.set_xticklabels([LETTER[f] for f in fams])
            if linear:
                ax.set_ylim(0, 1.05)
            else:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(l_label(l, d))
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=15, labelpad=8)
        y = y_plot + ph + xt
    handles = [Patch(facecolor=COLOR[f], label=f"{LETTER[f]} = {LABEL[f]}") for f in fams]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.0), ncol=ncol_leg, frameon=False,
               handlelength=1.5, fontsize=16)
    os.makedirs(a.outdir, exist_ok=True)
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{a.prefix}{stem}_main.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {a.prefix}{stem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/grad_quality.csv", help="summary CSV written by exp_grad_quality.py")
    ap.add_argument("--outdir", default=None, help="default: <csv dir>/figures")
    ap.add_argument("--functions", nargs="+", default=None, help="function names (rows); default: all in the CSV")
    ap.add_argument("--families", nargs="+", default=None, help="families to show; default: all in the CSV")
    ap.add_argument("--metrics", nargs="+", default=["err", "cover"], choices=list(METRICS))
    ap.add_argument("--d", type=int, default=None, help="dimension (default: the smallest in the CSV)")
    ap.add_argument("--dtype", default=None, help="float32 / float64 (default: the first in the CSV)")
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l to show (default: all)")
    ap.add_argument("--box", nargs=2, default=["p25", "p75"], choices=PCT, help="box limits")
    ap.add_argument("--band", nargs=2, default=["p05", "p95"], choices=PCT, help="whisker limits")
    ap.add_argument("--linear", nargs="*", default=[], choices=list(METRICS), help="metrics on a linear y axis")
    ap.add_argument("--floor", type=float, default=1e-16, help="values are clipped below this (log axes)")
    ap.add_argument("--prefix", default="", help="prefix for the output file names")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args()
    a.outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "figures")

    rows = read_rows(a.csv)
    if not rows:
        raise SystemExit(f"no rows in {a.csv}")
    present = list(dict.fromkeys(r["function"] for r in rows))
    functions = a.functions or present
    missing = [f for f in functions if f not in present]
    if missing:
        raise SystemExit(f"functions not in the CSV: {missing}  (available: {present})")
    d = a.d if a.d is not None else min(r["d"] for r in rows)
    dtype = a.dtype or rows[0]["dtype"]
    sub = [r for r in rows if r["d"] == d and r["dtype"] == dtype and r["function"] in functions]
    if not sub:
        raise SystemExit(f"no rows for d={d}, dtype={dtype}")
    fams_in = {r["family"] for r in sub}
    fams = [f for f in (a.families or ORDER) if f in fams_in]
    ls = sorted({r["l"] for r in sub}) if a.ls is None else [l for l in a.ls if l in {r["l"] for r in sub}]
    for metric in a.metrics:
        if f"{METRICS[metric][0]}_median" not in sub[0]:
            print(f"[skip] metric '{metric}': columns {METRICS[metric][0]}_* not in {a.csv}")
            continue
        make_figure(rows, metric, functions, fams, ls, d, dtype, a, linear=metric in a.linear)


if __name__ == "__main__":
    main()