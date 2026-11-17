import argparse
import os

import numpy as np
from matplotlib.lines import Line2D

from plot_common import ORDER, COLOR, MARKER, DTYPES, LABEL, NAN, plt, read_rows


FIGS = {   # key -> (part, x axis, file stem, title)
#    "ask_l": ("ask", "l", "exp2_ask_vs_l", "Ask: perturbed points, largest batch"),
#    "ask_batch": (None, "batch", "exp2_ask_vs_batch", r"Ask in batches, $\ell = d$"),
    "estimator_l": ("both", "l", "exp2_estimator_vs_l", " "),
}
MiB = 2.0 ** 20

# line style per family in the single-metric figures (same dashed curve as in plot_exp1)
LS = {"qr_f32_reorth": "--"}


def bottom_legend(fig, handles, fams, ncol=5):

    ordered = [f for f in fams if f in handles]
    if not ordered:
        return
    fig.legend([handles[f] for f in ordered], [LABEL[f] for f in ordered],   loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 0.1), mode="expand",
               borderaxespad=0.2, ncol=min(len(ordered), ncol), frameon=False,
               handlelength=2.5, fontsize=12)


def val(r, base, stat="median"):

    for k in (f"{base}_{stat}", base):
        v = r.get(k)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return NAN


def memory(r, cols):
    parts = [val(r, c, "median") for c in cols]
    if parts[0] != parts[0]:
        return NAN
    return sum(p for p in parts if p == p)


def points(rows, fam, d, dtype, part, mode):

    sel = [r for r in rows if r.get("part") == part and r["family"] == fam and r["d"] == d and r["dtype"] == dtype
           and r.get("status", "ok") == "ok" and r.get("method", "implicit") == "implicit"]
    if mode == "l":
        best = {}
        for r in sel:
            if r["l"] not in best or r["batch"] > best[r["l"]]["batch"]:
                best[r["l"]] = r
        return [(l, best[l]) for l in sorted(best)]
    return [(int(r["batch"]), r) for r in sorted((r for r in sel if r["l"] == d), key=lambda r: r["batch"])]


def make_figure(rows, key, metric, dtypes, dims, fams, a):

    part, mode, stem, title = FIGS[key]
    part = part or a.batch_part
    nr, nc = len(dtypes), len(dims)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc + 0.8, 3.1 * nr + 0.8), squeeze=False,
                            sharey="row" if a.sharey else False)
    any_data = False
    handles = {}                                   # family -> artist, for the legend
    for i, dtype in enumerate(dtypes):
        for j, d in enumerate(dims):
            ax = axs[i][j]
            panel_data = False
            for fam in fams:
                pts = points(rows, fam, d, dtype, part, mode)
                if not pts:
                    continue
                x = np.array([p[0] for p in pts], dtype=float)
                kw = dict(color=COLOR[fam], marker=MARKER[fam], ms=5, lw=2, ls=LS.get(fam, "-"), zorder=3)
                if metric == "time":
                    t = np.array([val(r, "time", "median") for _, r in pts])
                    lo = np.array([val(r, "time", a.band[0]) for _, r in pts])
                    hi = np.array([val(r, "time", a.band[1]) for _, r in pts])
                    err = None
                    if np.isfinite(lo).all() and np.isfinite(hi).all():
                        err = [np.maximum(t - lo, 0), np.maximum(hi - t, 0)]
                    h = ax.errorbar(x, t, yerr=err, capsize=3, elinewidth=1.2, capthick=1.2, **kw)
                    panel_data = panel_data or np.isfinite(t).any()
                else:
                    m = np.array([memory(r, a.mem_cols) for _, r in pts]) / MiB
                    h, = ax.plot(x, m, **kw)
                    panel_data = panel_data or np.isfinite(m).any()
                handles.setdefault(fam, h)
            any_data = any_data or panel_data
            ax.set_yscale("log")
            if mode == "batch":
                ax.set_xscale("log")
            if i ==0:
                ax.set_title(f"$d = {d}$", fontsize=16)
            if j == 0:      # dtype label of the row (same box as plot_exp1.py)
                ax.text(0.04, 0.95, dtype, transform=ax.transAxes, ha="left", va="top", fontsize=11, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.95))
                ax.set_ylabel("Time (s)" if metric == "time" else "Memory (MiB)", fontsize=16)
            if not panel_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            if i == nr - 1:
                ax.set_xlabel(r"$\ell$" if mode == "l" else "batch size", fontsize=16)
    if not any_data:
        print(f"[skip] {stem}_{metric}: no data with part='{part}'")
        plt.close(fig)
        return
    bottom_legend(fig, handles, fams)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    os.makedirs(a.outdir, exist_ok=True)
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{stem}_{metric}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {stem}_{metric}")


def make_batch_both_figure(rows, dtypes, dims, fams, a):

    part, mode, stem = "both", "batch", "exp2_estimator_vs_batch"
    nr, nc = len(dtypes), len(dims)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.1 * nr), squeeze=False,
                            sharey="row" if a.sharey else False)
    twins = [[None] * nc for _ in range(nr)]
    any_data = False
    handles = {}                                   # family -> artist, for the legend
    for i, dtype in enumerate(dtypes):
        for j, d in enumerate(dims):
            ax = axs[i][j]
            ax2 = twins[i][j] = ax.twinx()
            ax2.grid(False)
            ax.set_zorder(ax2.get_zorder() + 1)          # time curves drawn on top of the memory ones
            ax.patch.set_visible(False)
            panel_data = False
            for fam in fams:
                pts = points(rows, fam, d, dtype, part, mode)
                if not pts:
                    continue
                x = np.array([p[0] for p in pts], dtype=float)
                t = np.array([val(r, "time", "median") for _, r in pts])
                lo = np.array([val(r, "time", a.band[0]) for _, r in pts])
                hi = np.array([val(r, "time", a.band[1]) for _, r in pts])
                err = None
                if not a.no_errorbars and np.isfinite(lo).all() and np.isfinite(hi).all():
                    err = [np.maximum(t - lo, 0), np.maximum(hi - t, 0)]
                # time: solid, thin cap-less error bars so that the panel stays readable
                h = ax.errorbar(x, t, yerr=err, capsize=0, elinewidth=1.0, ecolor=COLOR[fam], color=COLOR[fam],
                                marker=MARKER[fam], ms=5, lw=2, zorder=3)
                handles.setdefault(fam, h)
                # memory: dashed, hollow markers, on the right axis
                m = np.array([memory(r, a.mem_cols) for _, r in pts]) / MiB
                ax2.plot(x, m, color=COLOR[fam], marker=MARKER[fam], ms=4, lw=1.4, ls="--", alpha=0.85,
                         markerfacecolor="white", markeredgecolor=COLOR[fam], zorder=2)
                panel_data = panel_data or bool(np.isfinite(t).any() or np.isfinite(m).any())
            any_data = any_data or panel_data
            for axx in (ax, ax2):
                axx.set_yscale("log")                     # x axis stays linear
            if a.sharey and j > 0:
                ax2.sharey(twins[i][0])
                ax2.tick_params(labelright=False)
            if i ==0:
                ax.set_title(f"$d = {d}$", fontsize=16)
            if j == 0:
                ax.text(0.04, 0.95, dtype, transform=ax.transAxes, ha="left", va="top", fontsize=11, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.95))
                ax.set_ylabel("Time (s)", fontsize=16)
            if j == nc - 1:
                ax2.set_ylabel("Memory (MiB)", fontsize=16)
            if not panel_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
            if i == nr - 1:
                ax.set_xlabel("batch size", fontsize=16)
    if not any_data:
        print(f"[skip] {stem}_time_memory: no data with part='{part}'")
        plt.close(fig)
        return
    style_handles = [Line2D([0], [0], color="black", lw=2, marker="o", ms=5, label="time (left axis)"),
                     Line2D([0], [0], color="black", lw=1.4, ls="--", marker="o", ms=4, markerfacecolor="white",
                            label="memory (right axis)")]
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=12, handlelength=2.5)
    bottom_legend(fig, handles, fams)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    os.makedirs(a.outdir, exist_ok=True)
    for ext in a.formats:
        fig.savefig(os.path.join(a.outdir, f"{stem}_time_memory.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {stem}_time_memory")


def fmt_bytes(b):
    if not (b == b) or b < 0:
        return "--"
    for unit, div in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if b >= div:
            return f"{b / div:.2f}~{unit}"
    return f"{int(b)}~B"


def tex(s):
    return str(s).replace("_", r"\_")


def table_env(header_lines, body, spec, caption, label):
    return (["\\begin{table}[ht]", "\\centering", "\\footnotesize", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{%s}" % spec, "\\toprule"] + header_lines + ["\\midrule"] + body +
            ["\\bottomrule", "\\end{tabular}}", "\\caption{%s}" % caption, "\\label{%s}" % label, "\\end{table}", ""])


def table_vs_l(rows, part, dtypes, dims, fams, a, title, label):

    head1 = "Method & " + " & ".join("\\multicolumn{%d}{c}{%s}" % (len(dims), dt) for dt in dtypes) + " \\\\"
    head2 = " & " + " & ".join(" & ".join(f"$d={d}$" for d in dims) for _ in dtypes) + " \\\\"
    body = []
    for fam in fams:
        cells = []
        for dt in dtypes:
            for d in dims:
                pts = points(rows, fam, d, dt, part, "l")
                if not pts:
                    cells.append("--")
                    continue
                _, r = min(pts, key=lambda p: abs(p[0] - a.frac * d))
                cells.append(fmt_bytes(memory(r, a.mem_cols)))
        body.append(f"{tex(LABEL[fam])} & " + " & ".join(cells) + " \\\\")
    spec = "l|" + "|".join("c" * len(dims) for _ in dtypes)
    cap = (f"{title}: GPU memory with $\\ell \\approx {a.frac:g}\\,d$ and the largest batch size "
           f"(columns: {tex(', '.join(a.mem_cols))}). `--' = not run.")
    return table_env([head1, head2], body, spec, cap, label)


def batch_label(b, l):
    if b == 1:
        return "$b=1$"
    for num, den, txt in ((1, 1, r"\ell"), (2, 3, r"2\ell/3"), (1, 2, r"\ell/2"), (1, 3, r"\ell/3")):
        if b == (num * l) // den:
            return f"$b={txt}$"
    return f"$b={b}$"


def table_vs_batch(rows, part, dtype, d, fams, a, label):

    sizes = sorted({int(b) for fam in fams for b, _ in points(rows, fam, d, dtype, part, "batch")})
    if not sizes:
        return []
    head = "Method & " + " & ".join(batch_label(b, d) for b in sizes) + " \\\\"
    body = []
    for fam in fams:
        by_b = {int(b): r for b, r in points(rows, fam, d, dtype, part, "batch")}
        cells = [fmt_bytes(memory(by_b[b], a.mem_cols)) if b in by_b else "--" for b in sizes]
        body.append(f"{tex(LABEL[fam])} & " + " & ".join(cells) + " \\\\")
    cap = (f"GPU memory of the {part} part in batches of size $b$ ($\\ell=d={d}$, {dtype}; "
           f"columns: {tex(', '.join(a.mem_cols))}). `--' = not run.")
    return table_env([head], body, "l|" + "c" * len(sizes), cap, f"{label}-{dtype}")


def write_tables(rows, dtypes, dims, fams, a):
    out = ["% LaTeX tables of the memory usage; needs \\usepackage{booktabs,graphicx}", ""]
    out += table_vs_l(rows, "ask", dtypes, dims, fams, a, "Ask (perturbed points)", "tab:exp2-memory-ask")
    out += table_vs_l(rows, "both", dtypes, dims, fams, a, "Entire estimator (ask + tell)", "tab:exp2-memory-estimator")
    d = a.table_d or max(dims)
    for dt in dtypes:
        out += table_vs_batch(rows, a.batch_part, dt, d, fams, a, "tab:exp2-memory-batch")
    path = os.path.join(a.outdir, "exp2_memory_tables.tex")
    os.makedirs(a.outdir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"[ok] exp2_memory_tables.tex -> {a.outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/exp2_action.csv")
    ap.add_argument("--outdir", default=None, help="default: <csv dir>/figures")
    ap.add_argument("--figures", nargs="+", default=list(FIGS), choices=list(FIGS))
    ap.add_argument("--batch-part", default="ask", choices=["ask", "tell", "both"],
                    help="part used by the ask_batch figure (default: ask)")
    ap.add_argument("--metrics", nargs="+", default=["time", "memory"], choices=["time", "memory"])
    ap.add_argument("--dtypes", nargs="+", default=None,
                    help="dtypes shown, one row each, in this order (default: all dtypes in the CSV, float64 first)")
    ap.add_argument("--dims", type=int, nargs="+", default=[500, 5000, 10000],
                    help="dimensions shown (default 500 5000 10000; those missing from the CSV are dropped)")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--band", nargs=2, default=["p05", "p95"], help="error bar of the time: lower and upper statistic")
    ap.add_argument("--mem-cols", nargs="+", default=["peak_bytes", "persistent_bytes"],
                    help="memory = sum of these columns (the first one must exist)")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    ap.add_argument("--no-errorbars", action="store_true",
                    help="time + memory batch figure: draw the median time without error bars (less crowded)")
    ap.add_argument("--frac", type=float, default=1.0, help="tables 1-2: use l closest to frac * d")
    ap.add_argument("--table-d", type=int, default=None, help="table 3: dimension (default: the largest)")
    ap.add_argument("--no-tables", action="store_true", help="do not write the LaTeX tables")
    a = ap.parse_args()
    a.outdir = a.outdir or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "figures")

    rows = read_rows(a.csv)
    if not rows:
        raise SystemExit(f"no rows in {a.csv}")
    if not any(val(r, "time", "median") == val(r, "time", "median") for r in rows):
        raise SystemExit(f"no time column (time_median / time) in {a.csv}; columns: {sorted(rows[0])}")
    if not any(memory(r, a.mem_cols) == memory(r, a.mem_cols) for r in rows):
        print(f"[warning] no values for the memory columns {a.mem_cols}; the memory panels will be empty. "
              f"Columns of the CSV: {sorted(rows[0])}")
    dtypes = a.dtypes or sorted({r["dtype"] for r in rows}, key=lambda t: (t != "float64", t))
    present = sorted({r["d"] for r in rows})
    dims = [d for d in a.dims if d in present]
    if not dims:
        print(f"[warning] none of the dimensions {a.dims} is in the CSV; using all of them: {present}")
        dims = present
    fams_in = {r["family"] for r in rows}
    fams = [f for f in (a.families or ORDER) if f in fams_in]
    for key in a.figures:
        for metric in a.metrics:
            make_figure(rows, key, metric, dtypes, dims, fams, a)
    # additional figure (ask + tell, l = d, x = batch size), time and memory together
    make_batch_both_figure(rows, dtypes, dims, fams, a)
    if not a.no_tables:
        write_tables(rows, dtypes, dims, fams, a)


if __name__ == "__main__":
    main()