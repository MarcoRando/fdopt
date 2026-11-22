import argparse
import csv
import datetime
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.append("../")
sys.path.append("../../")


from fdopt.directions.directions import FAMILIES
from fdopt.directions.directions import make as make_operator
from fdopt.optimizers.ffd import FFD
from fdopt.utils.linesearch import LineSearch
from exp_utils import (DTYPES, FAMILY_ORDER, FUNCTIONS, STATS, SEED, Writer, append_log, get_ls,
                          sample_point, summarize)
from synthetic_targets import (GriewankFunction, LeastSquares, QuingFunction, RegularizedLogistic, RosenbrockFunction, TridFunction)


# per-iteration metrics recorded by run_once
#   prog   : (F(x_k)-F*)/(F(x_0)-F*)
#   gamma  : line-search step size after iteration k
#   step   : accepted line-search step size gamma_k (0 if the search failed)
#   disp   : actual displacement gamma_k*||g_hat|| (comparable across direction families, unlike gamma_k alone)
#   fails  : cumulative number of step-size decreases (Armijo not satisfied)
#   lsfrac : cumulative share of the function evaluations spent in the line search
#   tpi    : line-search evaluations per outer iteration (running average)
#   btavg  : step-size decreases (backtracks) per outer iteration (running average)
BASE_METRICS = ["prog", "evals", "gamma", "step", "disp", "fails", "lsfrac", "tpi", "btavg"]
BASE_TRACE = ["evals", "prog", "gamma", "step", "disp", "fails", "lsfrac", "tpi", "btavg"]


def make_function(name, d, dtype, device, a):
    if name == "LeastSquares":
        return LeastSquares(d, mu=a.mu, L=a.L, set_l=True, dtype=dtype, device=device, seed=a.seed)
    if name == "RegularizedLogistic":
        return RegularizedLogistic(d, a.n, lam=a.lam, dtype=dtype, device=device, seed=a.seed)
    return {"Griewank": GriewankFunction, "Trid": TridFunction, "Rosenbrock": RosenbrockFunction,
            "Qing": QuingFunction}[name](d, dtype=dtype, device=device, seed=a.seed)


def make_ffd(fam, d, l, dtype, device, a):
    ls = LineSearch(gamma0=a.gamma0, c1=a.c1, beta=a.beta, gamma_min=a.gamma_min, expand=a.expand,
                    gamma_max=a.gamma_max, n_candidates=a.n_candidates, reset_on_fail=a.reset_on_fail)
    return FFD(make_operator(fam, d, l, dtype, device, seed=a.seed), h=a.h, linesearch=ls)


def run_once(ffd, f, x0, T):
    ls = ffd.ls
    ls.restart()
    ffd.n_evals = 0
    ffd.steps = []
    ffd.reset(x0.clone())
    fvals, evals, gammas, steps, disp, fails, lsc = [], [], [], [], [], [], []
    while True:
        while not ffd.done:                                  # gradient estimate AND line search of iteration k
            ffd.tell(f(ffd.ask()))
        fvals.append(ffd.fx.item())
        evals.append(ffd.n_evals)                            # includes the line search of iteration k
        gnorm = float(ffd.gradient().norm())
        ffd.step()
        steps.append(float(ffd.last_step))
        disp.append(steps[-1] * gnorm)                       # ||x_{k+1} - x_k||
        gammas.append(float(ls.gamma))
        fails.append(ls.n_fails)                              # cumulative number of step-size decreases
        lsc.append(ls.n_evals)                            # cumulative line-search evaluations (exact counter)
        if ffd.n_evals >= T:
            break
    ev, fa, lc = (np.array(v, dtype=np.float64) for v in (evals, fails, lsc))
    iters = np.arange(1, len(fvals) + 1, dtype=np.float64)
    out = {"f": np.array(fvals), "evals": ev, "gamma": np.array(gammas), "step": np.array(steps),
           "disp": np.array(disp), "fails": fa, "lsfrac": lc / ev, "tpi": lc / iters, "btavg": fa / iters}
    return out


def pad_stack(arrs):
    K = max(len(x) for x in arrs)
    out = np.full((len(arrs), K), np.nan)
    for i, x in enumerate(arrs):
        out[i, : len(x)] = x
    return out


def to_grid(mets, d, step, metrics):
    """Resample every run on a COMMON grid of function evaluations (multiples of step*(d+1)): the value at grid
    point g is the one of the last iteration that was completed with at most g evaluations (step function). Runs
    that already stopped are NaN. This makes the statistics across repetitions (and across the methods, which
    spend a different number of evaluations per iteration) comparable at equal evaluation budget."""
    evs = mets["evals"]
    gmax = max(e[-1] for e in evs) / (d + 1)
    grid = np.arange(step, gmax + 1e-9, step) * (d + 1)
    out = {m: [] for m in metrics}
    for i, ev in enumerate(evs):
        idx = np.searchsorted(ev, grid, side="right") - 1
        ok = (idx >= 0) & (grid <= ev[-1])
        for m in metrics:
            if m == "evals":
                v = np.where(ok, grid, np.nan)
            else:
                v = np.full(len(grid), np.nan)
                v[ok] = mets[m][i][idx[ok]]
            out[m].append(v)
    alive = np.isfinite(np.vstack(out["prog"])).any(axis=0)
    return {m: [v[alive] for v in out[m]] for m in metrics}


def stat_rows(mets, metrics, stride, base):
    """One row per stored iteration k: statistics across the repetitions of each metric (runs that already
    stopped are NaN-padded and ignored; n_reps = number of runs still alive at k)."""
    mats = {m: pad_stack(mets[m]) for m in metrics}
    K = mats["prog"].shape[1]
    ks = sorted(set(range(0, K, stride)) | {K - 1})
    rows = []
    for k in ks:
        row = dict(base, k=k, n_reps=int(np.isfinite(mats["prog"][:, k]).sum()))
        for m in metrics:
            v = mats[m][:, k]
            row.update(summarize(m, v))
            ok = v[np.isfinite(v)]          # mean and (sample) std, added if summarize does not provide them
            row.setdefault(f"{m}_mean", float(ok.mean()) if ok.size else float("nan"))
            row.setdefault(f"{m}_std", float(ok.std(ddof=1)) if ok.size > 1 else 0.0 if ok.size else float("nan"))
        rows.append(row)
    return rows


def write_trace(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_log(a, path, metrics, trace_metrics):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["Convergence experiment (FFD + Armijo line search)",
             f"started        : {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"command        : {' '.join(sys.argv)}",
             f"host           : {platform.node()} ({platform.platform()})",
             f"python / torch : {platform.python_version()} / {torch.__version__}",
             "",
             "Parameters",
             f"functions      : {', '.join(a.functions)}",
             f"families       : {', '.join(a.families)}",
             f"dtypes         : {', '.join(a.dtypes)}",
             f"dims           : {a.dims}",
             f"l (per d)      : " + "; ".join(f"d={d}: {get_ls(a, d)}" for d in a.dims),
             f"repetitions    : {a.reps} (different random x0 per repetition, shared by all methods)",
             f"budget T       : {a.T} function evaluations",
             f"metrics        : {', '.join(metrics)}",
             f"statistics     : {', '.join(STATS)}",
             f"summary csv    : {a.out}",
             f"trace dir      : {a.trace_dir} (metrics: {', '.join(trace_metrics)}; "
             + (f"common grid of {a.grid_step} x (d+1) evaluations" if not a.no_grid else f"per iteration, stride {a.trace_stride}") + ")",
             "",
             "All arguments"]
    lines += [f"  {k} = {v}" for k, v in sorted(vars(a).items())]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_block(fname, dname, d, a, summary_w, metrics, trace_metrics):
    dt, dev = DTYPES[dname], a.device
    f = make_function(fname, d, dt, dev, a)
    fmin = float(f.min_f)
    ls = get_ls(a, d)
    ffd = {(fam, l): make_ffd(fam, d, l, dt, dev, a) for fam in a.families for l in ls}
    res = {k: {m: [] for m in metrics} for k in ffd}
    xgen = torch.Generator(device=dev)
    for rep in range(a.reps):
        xgen.manual_seed(a.seed + 1000 * rep + 7)            # same x0 for every family and l
        x0 = sample_point(f, xgen, a.sample_scale)
        f0 = f(x0).reshape(-1)[0].item()
        denom = f0 - fmin
        if not np.isfinite(denom) or denom <= 0:
            continue
        for key, est in ffd.items():
            r = run_once(est, f, x0, a.T)
            r["prog"] = (r["f"] - fmin) / denom              # normalised function value (1 at x0, 0 at the minimum)
            for m in metrics:
                res[key][m].append(r[m])
        print(f"   rep {rep + 1}/{a.reps}", flush=True)

    # summary file (all the blocks in the same CSV), per iteration
    rows = []
    for (fam, l), mets in res.items():
        if not mets["prog"]:
            continue
        rows += stat_rows(mets, metrics, a.stride,
                          dict(function=fname, family=fam, dtype=dname, d=d, l=l, h=a.h))
    summary_w.write(rows)

    # trace files: one per method and target, statistics over the repetitions for every l
    if a.trace_dir:
        for fam in a.families:
            trows = []
            for l in ls:
                mets = res[(fam, l)]
                if not mets["prog"]:
                    continue
                if not a.no_grid:
                    mets = to_grid(mets, d, a.grid_step, trace_metrics)
                trows += stat_rows(mets, trace_metrics, 1 if not a.no_grid else a.trace_stride,
                                   dict(function=fname, family=fam, dtype=dname, d=d, l=l))
            if trows:
                path = os.path.join(a.trace_dir, f"trace_{fname}_{fam}_{dname}_d{d}.csv")
                write_trace(path, trows)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--functions", nargs="+", default=FUNCTIONS, choices=FUNCTIONS)
    ap.add_argument("--families", nargs="+", default=[f for f in FAMILY_ORDER if f in FAMILIES],
                    choices=list(FAMILIES))
    ap.add_argument("--dims", type=int, nargs="+", default=[50])
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l (default: 2, d/3, d/2, 4d/5, d)")
    ap.add_argument("--dtypes", nargs="+", default=["float64"], choices=list(DTYPES))
    ap.add_argument("--reps", type=int, default=50, help="number of random initialisations")
    ap.add_argument("--T", type=int, default=10000, help="budget of function evaluations per run")
    ap.add_argument("--h", type=float, default=1e-7, help="finite-difference step")
    ap.add_argument("--sample-scale", type=float, default=1.0, help="x0 ~ U(box) shrunk around the box centre")
    ap.add_argument("--stride", type=int, default=1,
                    help="summary csv: store every `stride`-th iteration (last one always kept)")
    # line search
    ap.add_argument("--gamma0", type=float, default=1.0)
    ap.add_argument("--c1", type=float, default=1e-7)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma-min", type=float, default=1e-10)
    ap.add_argument("--gamma-max", type=float, default=1000.0)
    ap.add_argument("--expand", type=float, default=2.0)
    ap.add_argument("--n-candidates", type=int, default=1)
    ap.add_argument("--no-reset-on-fail", dest="reset_on_fail", action="store_false")
    # targets
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=1e4)
    ap.add_argument("--n", type=int, default=1000, help="samples of RegularizedLogistic")
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/convergence_ls.csv")
    ap.add_argument("--trace-dir", default=None,
                    help="if given, write one CSV per (function, family, dtype, d) in this directory, with the "
                         "statistics over the repetitions of the traced metrics for every l")
    ap.add_argument("--no-grid", action="store_true",
                    help="traces per iteration (old behaviour) instead of on a common grid of evaluations")
    ap.add_argument("--grid-step", type=float, default=0.25,
                    help="trace grid spacing, in units of (d+1) function evaluations")
    ap.add_argument("--trace-stride", type=int, default=1,
                    help="per-iteration traces (--no-grid): store every `stride`-th step (last one always kept)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    metrics, trace_metrics = BASE_METRICS, BASE_TRACE

    log_path = a.log or os.path.splitext(a.out)[0] + ".log"
    write_log(a, log_path, metrics, trace_metrics)
    print(f"log -> {log_path}", flush=True)

    summary_w = Writer(a.out)
    t0 = time.time()
    for dname in a.dtypes:
        for d in a.dims:
            for fname in a.functions:
                run_block(fname, dname, d, a, summary_w, metrics, trace_metrics)
                msg = f"[ok] {fname:13s} {dname} d={d}  ({time.time() - t0:.0f}s)"
                print(msg, flush=True)
                append_log(log_path, msg)
    append_log(log_path, f"finished       : {datetime.datetime.now().isoformat(timespec='seconds')} "
                         f"(total {time.time() - t0:.0f}s)")
    print(f"summary -> {a.out}" + (f", traces -> {a.trace_dir}" if a.trace_dir else ""))


if __name__ == "__main__":
    main()