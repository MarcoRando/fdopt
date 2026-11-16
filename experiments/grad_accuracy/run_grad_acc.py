import argparse
import datetime
import os
import platform
import sys
import time

import torch

sys.path.append("../")
sys.path.append("../../")

from exp_utils import (DTYPES, FAMILY_ORDER, FUNCTIONS, STATS, SEED, Writer, append_log, coverage_error, default_log_path, get_ls, sample_point, summarize, true_grad)
from fdopt.directions.directions import FAMILIES
from fdopt.directions.directions import make as make_operator
from fdopt.optimizers.ffd import FFD
from synthetic_targets import (GriewankFunction, LeastSquares, QuingFunction, RegularizedLogistic, RosenbrockFunction, TridFunction)


LOGISTIC_N = 1000          # number of samples of the RegularizedLogistic problem


def make_function(name, d, dtype, device, mu, L):
    if name == "LeastSquares":
        return LeastSquares(d, mu=mu, L=L, dtype=dtype, device=device)
    if name == "RegularizedLogistic":
        return RegularizedLogistic(d, n=LOGISTIC_N, dtype=dtype, device=device)
    return {"Griewank": GriewankFunction, "Trid": TridFunction, "Rosenbrock": RosenbrockFunction,
            "Qing": QuingFunction}[name](d, dtype=dtype, device=device)


def write_log(a, path):

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dev = str(a.device)
    gpu = "-"
    if dev.startswith("cuda") and torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(torch.device(dev))
        except Exception:
            pass
    lines = [
        "Gradient-quality experiment",
        f"started        : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"command        : {' '.join(sys.argv)}",
        f"host           : {platform.node()} ({platform.platform()})",
        f"python / torch : {platform.python_version()} / {torch.__version__}",
        f"device         : {dev}   gpu: {gpu}",
        "",
        "Parameters",
        f"functions      : {', '.join(a.functions)}",
        f"families       : {', '.join(a.families)}",
        f"dtypes         : {', '.join(a.dtypes)}",
        f"dims           : {a.dims}",
        f"l (per d)      : " + "; ".join(f"d={d}: {get_ls(a, d)}" for d in a.dims),
        f"repetitions    : {a.reps}",
        f"h              : {a.h}",
        f"sample-scale   : {a.sample_scale}",
        f"seed           : {a.seed}",
        f"LeastSquares   : mu={a.mu}, L={a.L}",
        f"Logistic       : n={LOGISTIC_N}",
        "batch size     : one batch of l directions (batch_size=None)",
        "metrics        : err = ||g - grad|| / ||grad||   (approximation error);",
        "                 cover = ||grad - c P^T P grad|| / ||grad||, c = FFD.c   (subspace coverage error)",
        "statistics     : " + ", ".join(STATS),
        "",
        "Output",
        f"summary csv    : {a.out}",
        f"raw csv        : {a.raw_out}",
        f"log            : {path}",
        "",
        "All arguments",
    ]
    lines += [f"  {k} = {v}" for k, v in sorted(vars(a).items())]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def run_block(fname, dname, d, a, summary_w, raw_w):
    dt, dev = DTYPES[dname], a.device
    f = make_function(fname, d, dt, dev, a.mu, a.L)
    ls = get_ls(a, d)

    ffd = {(fam, l): FFD(make_operator(fam, d, l, dt, dev, seed=a.seed), h=a.h)   # batch_size=None -> one batch of l
           for fam in a.families for l in ls}
    err = {k: [] for k in ffd}
    cov = {k: [] for k in ffd}
    xgen = torch.Generator(device=dev)
    for rep in range(a.reps):
        xgen.manual_seed(a.seed + 1000 * rep + 7)               # same x for every family and l
        x = sample_point(f, xgen, a.sample_scale)
        gt = true_grad(f, x)
        n = torch.linalg.norm(gt)
        if not torch.isfinite(n) or n == 0:
            continue
        for (fam, l), est in ffd.items():
            g = est.estimate(f, x).reshape(-1)
            e = (torch.linalg.norm(g - gt) / n).item()
            ce = coverage_error(est.P, est.c, gt)
            err[(fam, l)].append(e)
            cov[(fam, l)].append(ce)
            if raw_w.path:
                raw_w.write([{"function": fname, "family": fam, "dtype": dname, "d": d, "l": l, "rep": rep, "h": a.h, "rel_err": e, "cover_err": ce}])
    summary_w.write([dict(function=fname, family=fam, dtype=dname, d=d, l=l, n_reps=len(err[(fam, l)]), h=a.h,
                          **summarize("err", err[(fam, l)]), **summarize("cover", cov[(fam, l)]))
                     for (fam, l) in ffd])


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--functions", nargs="+", default=FUNCTIONS, choices=FUNCTIONS)
    ap.add_argument("--families", nargs="+", default=[f for f in FAMILY_ORDER if f in FAMILIES],
                    choices=list(FAMILIES))
    ap.add_argument("--dims", type=int, nargs="+", default=[500])
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l (default: 2, d/3, d/2, 4d/5, d)")
    ap.add_argument("--dtypes", nargs="+", default=["float32"], choices=list(DTYPES))
    ap.add_argument("--reps", type=int, default=100, help="number of random points")
    ap.add_argument("--h", type=float, default=1e-5, help="finite-difference step")
    ap.add_argument("--sample-scale", type=float, default=1.0,
                    help="x ~ U(box) shrunk around the centre of f.bounds by this factor (1 = full standard box)")
    ap.add_argument("--mu", type=float, default=1.0, help="Least Squares strong convexity")
    ap.add_argument("--L", type=float, default=1e4, help="Least Squares smoothness (cond. number L/mu)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/grad_quality.csv")
    ap.add_argument("--raw-out", default=None, help="optional per-repetition CSV")
    ap.add_argument("--log", default=None, help="log file with the experiment parameters (default: <out>.log)")
    a = ap.parse_args()

    log_path = a.log or default_log_path(a.out)
    write_log(a, log_path)                                  # parameters are logged before anything runs
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