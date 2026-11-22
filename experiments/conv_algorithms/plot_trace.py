import argparse
import csv
import glob
import math
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_common import (COLOR, LABEL, LETTER, MARKER, ORDER, PALETTE, common_selection, l_label, plt,
                         read_rows)

try:                                                    # same style and font sizes as plot_grad_accuracy.py
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "grid.alpha": 0.5})

NAN = float("nan")
YLABEL = r"$\frac{F(x_k)-F^*}{F(x_0)-F^*}$"
XLABEL = r"$\#\ \mathrm{evals} / (d + 1)$"
METRICS = {   # key -> (column prefix, y label, file stem)
    "prog": ("prog", YLABEL, "conv_curves"),                                  # curves: normalised function value
    "gamma": ("gamma", r"$\gamma$", "gamma_curves"),                          # curves: line-search step size gamma
    "step_curve": ("step", r"accepted step $\gamma_k$", "step_curves"),         # curves: accepted line-search step
    "disp": ("disp", r"$\|x_{k+1}-x_k\|$", "disp_curves"),                    # curves: actual displacement (fair across families)
    "fails_curve": ("fails", "# of failures", "fails_curves"),                # curves: cumulative number of failures
    "lsfrac": ("lsfrac", "line-search share of evals", "lsfrac_curves"),      # curves: share of evals in the line search
    "tpi": ("tpi", "LS evals / iteration", "tpi_curves"),                     # curves: line-search evals per iteration
    "btavg": ("btavg", "backtracks / iteration", "backtrack_curves"),         # curves: step reductions per iteration
    "fails": ("fails", "# of failures", "fails_boxplot"),                     # boxplot: total number of failures
    "lsfrac_final": ("lsfrac", "line-search share of evals", "lsfrac_boxplot"),
    "tpi_final": ("tpi", "LS evals / iteration", "tpi_boxplot"),
    "step": ("step", r"accepted step $\gamma_k$", "step_boxplot"),                            # boxplots at STEP_TICKS
}
KIND = {k: "curves" for k in METRICS}
KIND.update({"fails": "final", "lsfrac_final": "final", "tpi_final": "final", "step": "step"})
CURVE_KEYS = [k for k in METRICS if KIND[k] == "curves"]
YLIM = {"lsfrac": (0, 1)}
XTICKS = [0, 5, 10, 15, 20]                             # fixed x ticks of the curves, in evaluations / (d + 1)
STEP_TICKS = [5, 10, 15, 20]                            # evaluations / (d + 1) at which a metric is tabulated
PCT = ["p05", "p25", "p75", "p95", "min", "max"]

# prefix -> (description for the captions, scale used in the tables, which value is the best (bold) or None)
FINAL = {
    "prog": (r"normalised function value $\frac{F(x_k)-F^*}{F(x_0)-F^*}$", 1, "min"),
    "gamma": (r"line-search step size $\gamma$", 1, None),
    "step": (r"accepted line-search step size $\gamma_k$", 1, None),
    "disp": (r"displacement $\|x_{k+1}-x_k\|=\gamma_k\|\hat g\|$", 1, None),
    "fails": ("cumulative number of line-search failures", 1, "min"),
    "lsfrac": (r"share of the function evaluations spent in the line search (\%)", 100, None),
    "tpi": ("line-search function evaluations per outer iteration", 1, None),
    "btavg": ("step-size reductions (backtracks) per outer iteration", 1, None),
}

fn_to_title = {
    'LeastSquares' : "Least Squares",
    'Qing' : "Qing",
    'Rosenbrock' : 'Rosenbrock',
    'RegularizedLogistic' : 'Logistic Regression',
    'Trid' : 'Trid',
    'Griewank' : 'Griewank'
}


# ---------------------------------------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------------------------------------
def read_path(path):
    paths = sorted(glob.glob(os.path.join(path, "trace_*.csv"))) if os.path.isdir(path) else [path]
    if not paths:
        raise SystemExit(f"no trace_*.csv files in {path}")
    rows = []
    for p in paths:
        rows += read_rows(p)
    return rows, paths


def load_rows(a):
    return read_path(a.trace_dir or a.csv)


def rows_of(sub, fn, fam, l, a):
    rs = [r for r in sub if r["function"] == fn and r["family"] == fam and r["l"] == l]
    if not rs:
        return []
    nmax = max(r["n_reps"] for r in rs)
    return sorted((r for r in rs if r["n_reps"] >= a.min_frac * nmax), key=lambda r: r["k"])


def curve(rs, d, prefix, a, log=True):
    # x is exact when the traces were written on the common evaluation grid (run_conv_exp.py default): every
    # method is compared at the same evaluation budget, whatever its number of evaluations per iteration.
    x = np.array([r["evals_median"] for r in rs], dtype=float) / (d + 1)   # evaluations per (d+1)
    y = [np.array([r[f"{prefix}_{s}"] for r in rs], dtype=float) for s in ("median", a.band[0], a.band[1])]
    ok = np.isfinite(x) & np.isfinite(y[0])
    if not ok.any():
        return None
    x, y = x[ok], [v[ok] for v in y]
    if prefix == "gamma" and a.gamma0 is not None:          # the step size used by the first iteration is gamma0
        x, y = np.concatenate([[0.0], x]), [np.concatenate([[a.gamma0], v]) for v in y]
    if log:
        y = [np.maximum(v, a.floor) for v in y]
    return (x, *y)


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


def value_final(rs, key, scale=1.0):
    """value of `key` in the last row of the trace (end of the runs)."""
    v = rs[-1].get(key, NAN) if rs else NAN
    return v * scale if isinstance(v, float) else NAN


def value_at(rs, key, tick, d, scale=1.0):
    r = step_at(rs, tick * (d + 1))
    v = r.get(key, NAN) if r is not None else NAN
    return v * scale if isinstance(v, float) else NAN


def have(sub, cols):
    return [c for c in cols if c not in sub[0]]


# ---------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------
def grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0, xlabel=XLABEL):

    nr, nc = len(functions), len(ls)
    left, w, gap, right = 1.5, 2.35, 0.72, 0.2          # left margin = tick labels + the fraction used as y label
    ph, xt, top = 2.0, 0.35, 0.1                        # plot height, room for the x tick labels, top margin
    ta = lambda i: 0.45 + (0.35 if i == 0 else 0.0)     # room above the plots: function name (+ l titles in row 0)
    W = left + nc * w + (nc - 1) * gap + right
    leg_h = 0.1 + 0.34 * math.ceil(len(handles) / 4)    # legend rows (4 columns)
    H = top + sum(ta(i) + ph + xt for i in range(nr)) + leg_h
    fig = plt.figure(figsize=tuple(a.figsize))
    y = top
    for i, fn in enumerate(functions):
        fig.text((left + (nc * w + (nc - 1) * gap) / 2) / W, 1 - (y + 0.05) / H, fn_to_title.get(fn, fn), ha="center",
                 va="top", fontsize=16, fontweight="bold")
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
            if i == nr - 1 and xlabel:
                ax.set_xlabel(xlabel, fontsize=14)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=20 if ylabel.startswith(r"$\frac") else 13, labelpad=8)
            elif a.sharey:
                ax.tick_params(labelleft=False)
        y = y_plot + ph + xt
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), ncol=4,
               frameon=False, handlelength=2.5, fontsize=16, mode='expand', borderaxespad=0.3)
    os.makedirs(a.outdir, exist_ok=True)
    tag = "-".join(functions) if a.functions else "all"
    stem = f"{a.prefix}{stem0}_{tag}_d{d}_{dtype}"
    fig.savefig(os.path.join(a.outdir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {stem}")


def style_curve_axis(ax, metric, log):
    ax.set_xlim(0, XTICKS[-1])
    ax.set_xticks(XTICKS)
    ax.set_yscale("log" if log else "linear")
    if metric in YLIM:
        ax.set_ylim(*YLIM[metric])


def make_curves(sub, metric, functions, fams, ls, d, dtype, a):
    prefix, ylabel, stem0 = METRICS[metric]
    log = metric not in a.linear
    ds = "default" if a.no_steps else "steps-post"

    def draw(ax, fn, l):
        any_curve = False
        for fam in fams:
            c = curve(rows_of(sub, fn, fam, l, a), d, prefix, a, log)
            if c is None:
                continue
            any_curve = True
            x, med, lo, hi = c
            ax.plot(x, med, color=COLOR[fam], lw=3, drawstyle=ds, rasterized=True)
            ax.fill_between(x, lo, hi, color=COLOR[fam], alpha=a.alpha, lw=0, step=None if a.no_steps else "post", rasterized=True)
        style_curve_axis(ax, metric, log)
        return any_curve

    handles = [Line2D([0], [0], color=COLOR[f], lw=3, label=LABEL[f]) for f in fams]
    grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0)


def make_final_boxplots(sub, metric, functions, fams, ls, d, dtype, a):
    """One box per family: distribution across the repetitions of the metric at the last row of the trace
    (e.g. the total cumulative number of failures at the end of the runs)."""
    prefix, ylabel, stem0 = METRICS[metric]

    def draw(ax, fn, l):
        stats, colors, pos = [], [], []
        for k, fam in enumerate(fams, start=1):
            rs = rows_of(sub, fn, fam, l, a)
            st = box_stats(rs[-1], prefix, a.box, a.band) if rs else None
            if st is not None:
                stats.append(st), colors.append(COLOR[fam]), pos.append(k)
        if stats:
            bp = ax.bxp(stats, positions=pos, widths=0.65, patch_artist=True, showfliers=False, manage_ticks=False,
                        medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2), boxprops=dict(lw=1.0))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c), patch.set_alpha(0.9)
        ax.set_xlim(0.4, len(fams) + 0.6)
        ax.set_xticks(range(1, len(fams) + 1))
        ax.set_xticklabels([LETTER[f] for f in fams])
        ax.set_ylim(bottom=0)
        return bool(stats)

    handles = [Patch(facecolor=COLOR[f], label=f"{LETTER[f]} = {LABEL[f]}") for f in fams]
    grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, stem0, xlabel=None)


def make_step_boxplots(sub, functions, fams, ls, d, dtype, a):
    """Step size at equal evaluation budgets (STEP_TICKS): the box is read at the grid point closest to the budget,
    so all the methods and values of l are compared after the same number of function evaluations."""
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


# ---------------------------------------------------------------------------------------------------------
# tables: one compact, readable table per target (rows = methods), no \resizebox, best value in bold
# ---------------------------------------------------------------------------------------------------------
def fmt_num(v, bold=False):
    if v is None or not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    e = int(e)
    if -2 <= e <= 2:
        s = f"{v:.3g}"
        return rf"\textbf{{{s}}}" if bold else s
    s = rf"{m}\cdot 10^{{{e}}}"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def tex(s):
    return str(s).replace("_", r"\_")


def title_of(fn):
    return tex(fn_to_title.get(fn, fn))


def tex_table(row_head, groups, subheads, rows, vals, best, caption, label):
    """groups: [(title, span)] spanning the subheads (or None); vals[i][j]: value of row i, column j (nan = '--');
    best: 'min' / 'max' / None -> bold the best value of every column."""
    nc = len(subheads)
    spans = [s for _, s in groups] if groups else [nc]
    spec = "l" + r"@{\hskip 10pt}".join("c" * s for s in spans)
    lines = ["\\begin{table}[ht]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{%s}" % spec, "\\toprule"]
    if groups:
        lines.append(f"{row_head} & " + " & ".join(f"\\multicolumn{{{s}}}{{c}}{{{t}}}" for t, s in groups) + " \\\\")
        c, cm = 2, ""
        for _, s in groups:
            cm += f"\\cmidrule(lr){{{c}-{c + s - 1}}}"
            c += s
        lines += [cm, " & " + " & ".join(subheads) + " \\\\"]
    else:
        lines.append(f"{row_head} & " + " & ".join(subheads) + " \\\\")
    lines.append("\\midrule")
    bests = []
    for j in range(nc):
        col = [r[j] for r in vals if r[j] is not None and np.isfinite(r[j])]
        bests.append((min(col) if best == "min" else max(col)) if (col and best) else None)
    for name, r in zip(rows, vals):
        cells = [fmt_num(v, bests[j] is not None and v == bests[j]) for j, v in enumerate(r)]
        lines.append(f"{tex(name)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\caption{%s}" % caption, "\\label{%s}" % label, "\\end{table}", ""]
    return lines


def save_tex(a, name, lines):
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{a.prefix}{name}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[ok] {name} -> {a.outdir}")


def tbl_ls(a, ls, narrow):
    """values of l shown in the tables: --table-ls, else (narrow) the smallest and largest, else all."""
    if a.table_ls:
        out = [l for l in a.table_ls if l in ls]
    elif narrow and len(ls) > 2:
        out = [ls[0], ls[-1]]
    else:
        out = ls
    return out or ls


def write_final_tables(sub, prefix, functions, fams, ls, d, dtype, a):
    """Median over the repetitions of the metric at the end of the runs; rows = methods, columns = l; one table
    per target."""
    desc, scale, best = FINAL[prefix]
    tl = tbl_ls(a, ls, narrow=False)
    lines = [f"% final {prefix} (median over the repetitions); needs \\usepackage{{booktabs}}", ""]
    for fn in functions:
        vals = [[value_final(rows_of(sub, fn, fam, l, a), f"{prefix}_median", scale) for l in tl] for fam in fams]
        cap = (f"{title_of(fn)}: median final {desc} at the end of the runs ($d={d}$, {dtype}) for different "
               f"numbers of directions $\\ell$. `--' = no data.")
        lines += tex_table("Method", None, [l_label(l, d) for l in tl], [LABEL[f] for f in fams], vals, best, cap,
                           f"tab:conv-final-{prefix}-{fn}")
    save_tex(a, f"final_{prefix}_tables_d{d}_{dtype}.tex", lines)


def write_time_tables(sub, prefix, functions, fams, ls, d, dtype, a):
    """Median of the metric after STEP_TICKS x (d+1) evaluations (same budget for every method); rows = methods,
    column groups = l, columns = budgets; one table per target."""
    desc, scale, best = FINAL[prefix]
    tl = tbl_ls(a, ls, narrow=True)
    ticks = ", ".join(str(t) for t in STEP_TICKS)
    groups = [(l_label(l, d), len(STEP_TICKS)) for l in tl]
    subheads = [str(t) for _ in tl for t in STEP_TICKS]
    lines = [f"% {prefix} at fixed evaluation budgets; needs \\usepackage{{booktabs}}", ""]
    for fn in functions:
        vals = [[value_at(rows_of(sub, fn, fam, l, a), f"{prefix}_median", t, d, scale)
                 for l in tl for t in STEP_TICKS] for fam in fams]
        cap = (f"{title_of(fn)}: median {desc} ($d={d}$, {dtype}) after $\\#\\,\\mathrm{{evals}}/(d+1) \\in "
               f"\\{{{ticks}\\}}$ function evaluations, for different $\\ell$. `--' = the runs stop before.")
        lines += tex_table("Method", groups, subheads, [LABEL[f] for f in fams], vals, best, cap,
                           f"tab:conv-time-{prefix}-{fn}")
    save_tex(a, f"time_{prefix}_tables_d{d}_{dtype}.tex", lines)


GAMMA_STATS = [("mean", "mean"), ("median", "median"), ("p05", "$p_{05}$"), ("p95", "$p_{95}$"),
               ("std", "std"), ("min", "min"), ("max", "max")]


def pick_l(ls, d):
    l = d if d in ls else max(ls)
    if l != d:
        print(f"[warning] l = d = {d} not in the data, using l = {l}")
    return l


def write_gamma_table(sub, functions, fams, ls, d, dtype, a):
    """Statistics over the repetitions of gamma at the end of the runs, for l = d; one table per target."""
    l = pick_l(ls, d)
    lines = ["% final step size gamma (l = d); needs \\usepackage{booktabs}", ""]
    for fn in functions:
        vals = [[value_final(rows_of(sub, fn, fam, l, a), f"gamma_{st}") for st, _ in GAMMA_STATS] for fam in fams]
        cap = (f"{title_of(fn)}: statistics over the repetitions of the line-search step size $\\gamma$ at the end of "
               f"the runs ($d={d}$, {dtype}, {l_label(l, d)}). `--' = no data.")
        lines += tex_table("Method", None, [h for _, h in GAMMA_STATS], [LABEL[f] for f in fams], vals, None, cap,
                           f"tab:conv-gamma-{fn}")
    save_tex(a, f"final_gamma_tables_d{d}_{dtype}.tex", lines)


LS_COLS = [(r"LS evals (\%)", "lsfrac_median", 100), ("LS evals / iter", "tpi_median", 1),
           ("backtracks / iter", "btavg_median", 1), ("failures", "fails_median", 1),
           (r"$\gamma$ median", "gamma_median", 1), (r"$\gamma$ $p_{05}$", "gamma_p05", 1),
           (r"$\gamma$ $p_{95}$", "gamma_p95", 1)]


def write_ls_summary(sub, functions, fams, ls, d, dtype, a):
    """Answer to 'which share of the evaluations goes to the line search, how many trials per iteration, what is
    gamma': one table per target, rows = methods, everything at the end of the runs, for l = d."""
    l = pick_l(ls, d)
    cols = [c for c in LS_COLS if c[1] in sub[0]]
    lines = ["% line-search summary (l = d); needs \\usepackage{booktabs}", ""]
    for fn in functions:
        vals = [[value_final(rows_of(sub, fn, fam, l, a), key, sc) for _, key, sc in cols] for fam in fams]
        cap = (f"{title_of(fn)}: line-search summary at the end of the runs ($d={d}$, {dtype}, {l_label(l, d)}; "
               f"median over the repetitions). LS evals = share of all function evaluations spent in the line search; "
               f"LS evals / iter = line-search evaluations per outer iteration; backtracks = step-size reductions "
               f"per outer iteration.")
        lines += tex_table("Method", None, [h for h, _, _ in cols], [LABEL[f] for f in fams], vals, None, cap,
                           f"tab:ls-summary-{fn}")
    save_tex(a, f"ls_summary_tables_d{d}_{dtype}.tex", lines)


# ---------------------------------------------------------------------------------------------------------
# ablation of the line search (variants = runs with other line-search settings: c1, beta, expand, gamma0, reset_on_fail)
# ---------------------------------------------------------------------------------------------------------
def make_variant_curves(variants, metric, fam, functions, ls, d, dtype, a):
    prefix, ylabel, stem0 = METRICS[metric]
    log = metric not in a.linear
    ds = "default" if a.no_steps else "steps-post"

    def draw(ax, fn, l):
        any_curve = False
        for i, (_, vsub) in enumerate(variants):
            c = curve(rows_of(vsub, fn, fam, l, a), d, prefix, a, log)
            if c is None:
                continue
            any_curve = True
            x, med, lo, hi = c
            ax.plot(x, med, color=PALETTE[i % len(PALETTE)], lw=3, drawstyle=ds, rasterized=True)
            ax.fill_between(x, lo, hi, color=PALETTE[i % len(PALETTE)], alpha=a.alpha, lw=0,
                            step=None if a.no_steps else "post", rasterized=True)
        style_curve_axis(ax, metric, log)
        return any_curve

    handles = [Line2D([0], [0], color=PALETTE[i % len(PALETTE)], lw=3, label=lab) for i, (lab, _) in enumerate(variants)]
    grid_figure(functions, ls, d, dtype, a, draw, ylabel, handles, f"ablation_{stem0}_{fam}")


def write_variant_tables(variants, fam, functions, ls, d, dtype, a):
    """rows = variants, columns = targets (l = d); one table per metric, for one method."""
    l = pick_l(ls, d)
    lines = [f"% line-search ablation, method {fam}; needs \\usepackage{{booktabs}}", ""]
    for prefix in a.variant_tables:
        desc, scale, best = FINAL[prefix]
        vals = [[value_final(rows_of(vsub, fn, fam, l, a), f"{prefix}_median", scale) for fn in functions]
                for _, vsub in variants]
        if not np.isfinite(np.array(vals, dtype=float)).any():
            continue
        cap = (f"Line-search ablation for the {tex(LABEL[fam])} directions ($d={d}$, {dtype}, {l_label(l, d)}): "
               f"median final {desc}.")
        lines += tex_table("Variant", None, [title_of(fn) for fn in functions], [lab for lab, _ in variants], vals, best,
                           cap, f"tab:ablation-{prefix}-{fam}")
    save_tex(a, f"variants_{fam}_d{d}_{dtype}.tex", lines)


# ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/convergence_ls.csv", help="summary CSV written by run_conv_exp.py")
    ap.add_argument("--trace-dir", default=None,
                    help="directory with the trace_*.csv files (run_conv_exp.py --trace-dir); used instead of --csv")
    ap.add_argument("--outdir", default=None, help="default: <csv or trace dir>/figures")
    ap.add_argument("--functions", nargs="+", default=None, help="targets to plot (rows); default: all in the data")
    ap.add_argument("--families", nargs="+", default=None, help="families to show; default: all in the data")
    ap.add_argument("--metrics", nargs="+", default=["prog", "gamma", "step_curve", "disp", "fails_curve", "fails", "lsfrac",
                                                     "tpi", "lsfrac_final", "step"],
                    choices=list(METRICS),
                    help="curves: prog (normalised F), gamma, step_curve (step taken), fails_curve (cumulative "
                         "failures), lsfrac (share of evals in the line search), tpi (LS evals / iteration), btavg "
                         "(backtracks / iteration); "
                         "boxplots of the last row: fails, lsfrac_final, tpi_final; "
                         "step: step-size boxplots at 5, 10, 15, 20 evals/(d+1). "
                         "Metrics whose columns are not in the traces are skipped.")
    ap.add_argument("--linear", nargs="*", default=["fails_curve", "lsfrac", "tpi", "btavg"], choices=CURVE_KEYS,
                    help="curve metrics on a linear y axis (the others are log)")
    ap.add_argument("--d", type=int, default=None, help="dimension (default: the smallest in the data)")
    ap.add_argument("--dtype", default=None, help="float32 / float64 (default: the first in the data)")
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l to show (default: all)")
    ap.add_argument("--band", nargs=2, default=["p25", "p75"], choices=PCT,
                    help="shaded band of the curves / error bars (whiskers) of the boxplots")
    ap.add_argument("--box", nargs=2, default=["p25", "p75"], choices=PCT, help="box limits of the boxplots")
    ap.add_argument("--alpha", type=float, default=0.2, help="opacity of the band")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="drop the final steps reached by fewer than this fraction of the runs "
                         "(the 'last row' of the boxplots is the last one that remains)")
    ap.add_argument("--no-steps", action="store_true", help="join the points with straight lines (default: steps)")
    ap.add_argument("--sharey", action="store_true", help="share the y axis along each row")
    # tables
    ap.add_argument("--no-tables", action="store_true", help="do not write the LaTeX tables")
    ap.add_argument("--table-metrics", nargs="*", default=["prog", "fails", "lsfrac", "tpi", "btavg", "gamma"],
                    choices=list(FINAL), help="tables of the final value (rows = methods, columns = l, one table per "
                                              "target); 'gamma' gives mean/median/p05/p95/std/min/max for l = d")
    ap.add_argument("--time-tables", nargs="*", default=["step", "disp", "gamma", "lsfrac", "prog"], choices=list(FINAL),
                    help="tables of a metric at 5, 10, 15, 20 evals/(d+1) (rows = methods, column groups = l)")
    ap.add_argument("--table-ls", type=int, nargs="*", default=None,
                    help="values of l in the tables (default: all for the final tables, the smallest and the "
                         "largest for the tables at fixed budgets)")
    # line-search ablation
    ap.add_argument("--variants", nargs="*", default=None,
                    help="LABEL=PATH pairs (trace dir or csv) of runs with other line-search settings, e.g. "
                         "default=traces_default c1_1e-3=traces_c1 beta_0.8=traces_beta; the main data is NOT added automatically")
    ap.add_argument("--variant-families", nargs="*", default=None, help="methods shown in the ablation (default: all)")
    ap.add_argument("--variant-metrics", nargs="*", default=["prog", "gamma", "lsfrac"], choices=CURVE_KEYS)
    ap.add_argument("--variant-tables", nargs="*", default=["prog", "lsfrac", "tpi", "gamma"], choices=list(FINAL))
    ap.add_argument("--gamma0", type=float, default=1.0,
                    help="first step size, drawn at 0 evaluations in the gamma curves (use a negative value to omit)")
    ap.add_argument("--figsize", type=float, nargs=2, default=[15, 11], help="figure size in inches (as plot_grad_accuracy)")
    ap.add_argument("--floor", type=float, default=1e-16, help="values are clipped below this (log axis)")
    ap.add_argument("--prefix", default="", help="prefix for the output file names")
    a = ap.parse_args()
    if a.gamma0 < 0:
        a.gamma0 = None
    a.outdir = a.outdir or os.path.join(os.path.abspath(a.trace_dir or os.path.dirname(os.path.abspath(a.csv))),
                                        "figures")

    rows, paths = load_rows(a)
    if not rows:
        raise SystemExit(f"no rows in {paths}")
    functions, d, dtype, fams, ls, sub = common_selection(rows, a.functions, a.d, a.dtype, a.families, a.ls)

    # ---- figures of the traces
    for metric in a.metrics:
        prefix = METRICS[metric][0]
        need = ["evals_median", f"{prefix}_median", f"{prefix}_{a.band[0]}", f"{prefix}_{a.band[1]}"]
        if KIND[metric] in ("step", "final"):
            need += [f"{prefix}_{a.box[0]}", f"{prefix}_{a.box[1]}"]
        miss = have(sub, need)
        if miss:
            print(f"[skip] metric '{metric}': columns {miss} not found "
                  f"(trace files need the '{prefix}' statistics; available: {sorted(sub[0])})")
            continue
        if KIND[metric] == "step":
            make_step_boxplots(sub, functions, fams, ls, d, dtype, a)
        elif KIND[metric] == "final":
            make_final_boxplots(sub, metric, functions, fams, ls, d, dtype, a)
        else:
            make_curves(sub, metric, functions, fams, ls, d, dtype, a)

    # ---- tables
    if not a.no_tables:
        for m in a.table_metrics:
            need = [f"gamma_{st}" for st, _ in GAMMA_STATS] if m == "gamma" else [f"{m}_median"]
            miss = have(sub, need)
            if miss:
                print(f"[skip] table '{m}': columns {miss} not found")
            elif m == "gamma":
                write_gamma_table(sub, functions, fams, ls, d, dtype, a)
            else:
                write_final_tables(sub, m, functions, fams, ls, d, dtype, a)
        for m in a.time_tables:
            miss = have(sub, [f"{m}_median"])
            if miss:
                print(f"[skip] time table '{m}': columns {miss} not found")
            else:
                write_time_tables(sub, m, functions, fams, ls, d, dtype, a)
        if not have(sub, ["lsfrac_median", "gamma_median"]):
            write_ls_summary(sub, functions, fams, ls, d, dtype, a)

    # ---- line-search ablation
    if a.variants:
        variants = []
        for spec in a.variants:
            lab, _, path = spec.partition("=")
            vrows, _ = read_path(path)
            variants.append((lab, common_selection(vrows, functions, d, dtype, fams, ls)[-1]))
        vfams = [f for f in (a.variant_families or fams)
                 if all(any(r["family"] == f for r in s) for _, s in variants)]
        for fam in vfams:
            for m in a.variant_metrics:
                prefix = METRICS[m][0]
                if have(variants[0][1], [f"{prefix}_median", f"{prefix}_{a.band[0]}", f"{prefix}_{a.band[1]}"]):
                    print(f"[skip] ablation metric '{m}': columns not found")
                    continue
                make_variant_curves(variants, m, fam, functions, ls, d, dtype, a)
            if not a.no_tables:
                write_variant_tables(variants, fam, functions, ls, d, dtype, a)


if __name__ == "__main__":
    main()