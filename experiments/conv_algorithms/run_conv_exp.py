import argparse
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


METRICS = ["prog", "evals", "gamma", "step", "fails"]


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

    ffd.ls.restart()
    ffd.n_evals = 0
    ffd.steps = []
    ffd.reset(x0.clone())
    fvals, evals, gammas, steps, fails = [], [], [], [], []
    nfail = 0
    while True:
        while not ffd.done:
            ffd.tell(f(ffd.ask()))
        fvals.append(ffd.fx.item())
        evals.append(ffd.n_evals)
        ffd.step()
        nfail += int(not ffd.ls.accepted)
        steps.append(float(ffd.last_step))
        gammas.append(float(ffd.ls.gamma))
        fails.append(nfail)
        if ffd.n_evals >= T:
            break
    return {"f": np.array(fvals), "evals": np.array(evals, dtype=np.float64), "gamma": np.array(gammas),
            "step": np.array(steps), "fails": np.array(fails, dtype=np.float64)}


def pad_stack(arrs):
    K = max(len(x) for x in arrs)
    out = np.full((len(arrs), K), np.nan)
    for i, x in enumerate(arrs):
        out[i, : len(x)] = x
    return out


def write_log(a, path):
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
             f"metrics        : {', '.join(METRICS)}",
             f"statistics     : {', '.join(STATS)}",
             f"summary csv    : {a.out}",
             f"raw csv        : {a.raw_out}",
             "",
             "All arguments"]
    lines += [f"  {k} = {v}" for k, v in sorted(vars(a).items())]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_block(fname, dname, d, a, summary_w, raw_w):
    dt, dev = DTYPES[dname], a.device
    f = make_function(fname, d, dt, dev, a)
    fmin = float(f.min_f)
    ls = get_ls(a, d)
    ffd = {(fam, l): make_ffd(fam, d, l, dt, dev, a) for fam in a.families for l in ls}
    res = {k: {m: [] for m in METRICS} for k in ffd}
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
            r["prog"] = (r["f"] - fmin) / denom
#            print(f"[--] feval: {r['prog']}")
            for m in METRICS:
                res[key][m].append(r[m])
            if raw_w.path:
                raw_w.write([{"function": fname, "family": key[0], "dtype": dname, "d": d, "l": key[1], "rep": rep,
                              "k": k, "f": r["f"][k], **{m: r[m][k] for m in METRICS}}
                             for k in range(0, len(r["f"]), a.stride)])
        print(f"   rep {rep + 1}/{a.reps}", flush=True)
    rows = []
    for (fam, l), mets in res.items():
        if not mets["prog"]:
            continue
        mats = {m: pad_stack(mets[m]) for m in METRICS}
        K = mats["prog"].shape[1]
        ks = sorted(set(range(0, K, a.stride)) | {K - 1})
        for k in ks:
            row = dict(function=fname, family=fam, dtype=dname, d=d, l=l, k=k, h=a.h,
                       n_reps=int(np.isfinite(mats["prog"][:, k]).sum()))
            for m in METRICS:
                row.update(summarize(m, mats[m][:, k]))
            rows.append(row)
    summary_w.write(rows)


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
    ap.add_argument("--stride", type=int, default=1, help="store every `stride`-th iteration (last one always kept)")
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
    ap.add_argument("--raw-out", default=None, help="optional per-repetition, per-iteration CSV (can be large)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    log_path = a.log or os.path.splitext(a.out)[0] + ".log"
    write_log(a, log_path)
    print(f"log -> {log_path}", flush=True)

    summary_w, raw_w = Writer(a.out), Writer(a.raw_out)
    t0 = time.time()
    for dname in a.dtypes:
        for d in a.dims:
            for fname in a.functions:
                run_block(fname, dname, d, a, summary_w, raw_w)
                msg = f"[ok] {fname:13s} {dname} d={d}  ({time.time() - t0:.0f}s)"
                print(msg, flush=True)
                append_log(log_path, msg)
    append_log(log_path, f"finished       : {datetime.datetime.now().isoformat(timespec='seconds')} "
                         f"(total {time.time() - t0:.0f}s)")
    print(f"summary -> {a.out}" + (f", raw -> {a.raw_out}" if a.raw_out else ""))


if __name__ == "__main__":
    main()