"""Plot and tabulate Experiment 1 (generation of the explicit direction matrix). Needs only matplotlib.

Figure : exp1_generation_time.{png,pdf}   rows = dtype (float64, float32), columns = dimension d,
         x axis = l (linear), y axis = time (log), curves = families: the median generation time as a point
         joined by a line, with error bars from the lower to the upper percentile (default 5th-95th, the
         smallest/largest stored by the experiment; change with --band).
Tables : exp1_memory_table.tex   memory footprint per family = peak while building the matrix + persistent
         exp1_time_table.tex     statistics of the time (mean, std, median, percentiles) per family
         (tables use the entry with l closest to frac * d, default l = d; change with --frac)

  python plot_exp1.py
  python plot_exp1.py --exclude qr_reorth qr_f32_reorth --band p25 p75 --frac 0.5
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results_utils import FIGURES_DIR, RESULTS_DIR, read_csv  # noqa: E402

NAN = float("nan")
ORDER = ["coordinate", "qr", "qr_reorth", "qr_f32_reorth", "householder", "permuted_householder", "butterfly",
         "gaussian", "spherical", "rademacher"]
# seaborn "colorblind" palette (hard-coded so that seaborn itself is not needed) + one marker per family
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]
COLOR = {f: PALETTE[i] for i, f in enumerate(ORDER)}
MARKER = {f: m for f, m in zip(ORDER, "osD^vP*X<>")}
DTYPES = ["float64", "float32"]
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 12, "grid.alpha": 0.5})


def val(r, base, stat="median"):
    """Value of a measurement: <base>_<stat> if the CSV has summary columns, else the scalar column <base>."""
    for k in (f"{base}_{stat}", base):
        v = r.get(k)
        if isinstance(v, (int, float)) and v == v:
            return v
    return NAN


def tex(s):
    return str(s).replace("_", r"\_")


def fmt_bytes(b):
    if not (b == b) or b < 0:
        return "--"
    for unit, div in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if b >= div:
            return f"{b / div:.2f}~{unit}"
    return f"{int(b)}~B"


def footprint(r):
    """Peak extra memory while building + memory held by the operator object (NaN if memory was not measured)."""
    p = val(r, "peak_bytes")
    if not (p == p):
        return NAN
    q = val(r, "persistent_bytes")
    return p + (q if q == q else 0.0)


def fmt_ms(t):
    return "--" if not (t == t) else f"{1e3 * t:.3g}"


def order(fams):
    return sorted(fams, key=lambda f: ORDER.index(f) if f in ORDER else len(ORDER))


def closest(rows, d, frac):
    """Row with l closest to frac * d (None if rows is empty)."""
    return min(rows, key=lambda r: abs(r["l"] - frac * d)) if rows else None


def make_figure(rows, dims, fams, a):
    nr, nc = len(DTYPES), len(dims)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.1 * nr), squeeze=False, sharey="row" if a.sharey else False)
    lo_s, hi_s = a.band
    for i, dt in enumerate(DTYPES):
        for j, d in enumerate(dims):
            ax = axs[i][j]
            sel = [r for r in rows if r["dtype"] == dt and r["d"] == d]
            for fam in fams:
                rs = sorted([r for r in sel if r["family"] == fam], key=lambda r: r["l"])
                if not rs:
                    continue
                x = [r["l"] for r in rs]
                med = [val(r, "time", "median") for r in rs]
                lo = [max(m - val(r, "time", lo_s), 0.0) for m, r in zip(med, rs)]   # error-bar lengths below / above
                hi = [max(val(r, "time", hi_s) - m, 0.0) for m, r in zip(med, rs)]
                ax.errorbar(x, med, yerr=[lo, hi], marker=MARKER.get(fam, "o"), ms=5, lw=2, capsize=3, elinewidth=1.3,
                            capthick=1.3, color=COLOR.get(fam), label=fam, zorder=3)
            ax.set(yscale="log", title=f"$d = {d}$")      # x axis stays linear
            if j == 0:     # dtype label in a small box inside the first subplot of the row
                ax.text(0.04, 0.95, dt, transform=ax.transAxes, ha="left", va="top", fontsize=11, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.95))
            if not sel:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            if i == nr - 1:
                ax.set_xlabel(r"$\ell$", fontsize=16)
            if j == 0:
                ax.set_ylabel("Time (s)", fontsize=16)
    handles = {}
    for ax in axs.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=min(len(handles), 5), frameon=False,
               handlelength=2.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    os.makedirs(a.outdir, exist_ok=True)
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"exp1_generation_time.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("[ok] exp1_generation_time")


def memory_table(rows, dims, fams, a):
    dts = [dt for dt in DTYPES if any(r["dtype"] == dt for r in rows)]
    lines = [r"\begin{table}[ht]", r"\centering", r"\footnotesize", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "|".join("c" * len(dims) for _ in dts).join(["|", ""]) + "}", r"\toprule",
             "Method & " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(dims), dt) for dt in dts) + r" \\"]
    lines.append(" & " + " & ".join(" & ".join(f"$d={d}$" for d in dims) for _ in dts) + r" \\")
    lines.append(r"\midrule")
    for fam in fams:
        cells = []
        for dt in dts:
            for d in dims:
                r = closest([x for x in rows if x["family"] == fam and x["dtype"] == dt and x["d"] == d], d, a.frac)
                if r is None:
                    cells.append("--")
                else:
                    cells.append(fmt_bytes(footprint(r)))
        lines.append(f"{tex(fam)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{GPU memory footprint of generating the explicit matrix with $l \approx %s\,d$ (measured once): "
              r"peak extra memory while building the matrix plus the memory held by the operator object between "
              r"calls (e.g.\ a stored identity matrix). `--' = not run.}" % f"{a.frac:g}",
              r"\label{tab:exp1-memory}", r"\end{table}"]
    path = os.path.join(a.outdir, "exp1_memory_table.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("[ok] exp1_memory_table.tex")


def time_table(rows, dims, fams, a):
    cols = [("mean", "mean"), ("std", "std"), ("median", "median"), ("p05", "p05"), ("p25", "p25"), ("p75", "p75"),
            ("p95", "p95")]
    lines = [r"\begin{longtable}{llr" + "r" * len(cols) + "}",
             r"\caption{Generation time (ms) of the explicit matrix with $l \approx %s\,d$ over the timed repetitions: "
             r"mean, standard deviation, median and percentiles.}\label{tab:exp1-time}\\" % f"{a.frac:g}",
             r"\toprule", "dtype & method & $l$ & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule", r"\endfirsthead",
             r"\toprule", "dtype & method & $l$ & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    first = True
    for dt in DTYPES:
        for d in dims:
            block = []
            for fam in fams:
                r = closest([x for x in rows if x["family"] == fam and x["dtype"] == dt and x["d"] == d], d, a.frac)
                if r is not None:
                    block.append(f"{dt} & {tex(fam)} & {r['l']} & " + " & ".join(fmt_ms(val(r, "time", c[0])) for c in cols) + r" \\")
            if block:
                if not first:
                    lines.append(r"\midrule")
                lines.append(r"\multicolumn{%d}{l}{\textit{$d=%d$, %s}} \\" % (3 + len(cols), d, dt))
                lines += block
                first = False
    lines.append(r"\end{longtable}")
    path = os.path.join(a.outdir, "exp1_time_table.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("[ok] exp1_time_table.tex  (needs \\usepackage{booktabs,longtable})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=RESULTS_DIR)
    ap.add_argument("--csv", default="exp1_generation.csv")
    ap.add_argument("--outdir", default=FIGURES_DIR)
    ap.add_argument("--exclude", nargs="*", default=["qr_reorth"], help="families to leave out")
    ap.add_argument("--band", nargs=2, default=["p05", "p95"], choices=["p05", "p25", "p75", "p95", "min", "max"],
                    help="lower and upper end of the shaded band")
    ap.add_argument("--dims", type=int, nargs="*", default=None, help="dimensions to show (default: all in the file)")
    ap.add_argument("--frac", type=float, default=1.0, help="tables: use l closest to frac * d")
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args()

    path = os.path.join(a.indir, a.csv)
    rows = [r for r in read_csv(path) if r.get("status", "ok") == "ok" and r["family"] not in a.exclude]
    if not rows:
        raise SystemExit(f"no usable rows in {path}")
    dims = sorted({r["d"] for r in rows}) if a.dims is None else a.dims
    fams = order({r["family"] for r in rows})
    make_figure(rows, dims, fams, a)
    memory_table(rows, dims, fams, a)
    time_table(rows, dims, fams, a)


if __name__ == "__main__":
    main()