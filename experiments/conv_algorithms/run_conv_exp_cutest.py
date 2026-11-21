import argparse
import datetime
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.append("../")
sys.path.append("../../")


from exp_utils import (DTYPES, FAMILY_ORDER, STATS, SEED, Writer, append_log, default_log_path, get_ls, summarize)
from fdopt.directions.directions import FAMILIES
from run_convergence_ls import METRICS, make_ffd, pad_stack, run_once
from run_grad_acc_cutest import load_problem, parse_sif_params, sample_points, select_problems
from synthetic_targets import PycutestTarget



def write_log(a, path, names):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["Convergence experiment (FFD + Armijo line search) on CUTEst",
             f"started        : {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"command        : {' '.join(sys.argv)}",
             f"host           : {platform.node()} ({platform.platform()})",
             f"python / torch : {platform.python_version()} / {torch.__version__}",
             "",
             "Parameters",
             f"problems       : {', '.join(names)}",
             f"sif params     : {a.sif_params}",
             f"families       : {', '.join(a.families)}",
             f"dtypes         : {', '.join(a.dtypes)}",
             f"l              : {a.ls if a.ls else '2, d/3, d/2, 4d/5, d (per problem)'}",
             f"repetitions    : {a.reps} (different random x0 per repetition, shared by all methods)",
             f"x0             : x0 + sigma (1+|x0|) N(0,I), sigma={a.sample_scale}, clipped to bounds",
             f"budget T       : {a.T} function evaluations",
             f"f* (fmin)      : {a.fmin}",
             f"metrics        : {', '.join(METRICS)}",
             f"statistics     : {', '.join(STATS)}",
             f"summary csv    : {a.out}",
             f"raw csv        : {a.raw_out}",
             "",
             "All arguments"]
    lines += [f"  {k} = {v}" for k, v in sorted(vars(a).items())]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def get_fmin(pname, a, fstars, runs):
    """f* of a problem according to --fmin; `runs` is the list of all f trajectories (for `best`)."""
    if a.fmin == "zero":
        return 0.0
    if fstars is not None and pname in fstars:
        return float(fstars[pname])
    vals = [np.nanmin(r) for r in runs if len(r) and np.isfinite(r).any()]
    return float(min(vals)) if vals else float("nan")


def run_block(pname, dname, a, fstars, summary_w, raw_w):
    dt, dev = DTYPES[dname], a.device
    prob = load_problem(pname, a.sif_params)
    f = PycutestTarget(prob, dtype=dt, device=dev, seed=a.seed)
    d = f.d
    if d < 2:
        print(f"   [skip] {pname}: d={d}", flush=True)
        f.close()
        return
    ls = get_ls(a, d)
    ffd = {(fam, l): make_ffd(fam, d, l, dt, dev, a) for fam in a.families for l in ls}
    res = {k: {m: [] for m in METRICS + ["f"]} for k in ffd}
    f0s, rep_ids = [], []
    xgen = torch.Generator(device=dev)
    try:
        for rep in range(a.reps):
            xgen.manual_seed(a.seed + 1000 * rep + 7)            # same x0 for every family and l
            x0 = sample_points(prob, f, xgen, a.sample_scale)
            f0 = f(x0).reshape(-1)[0].item()
            if not np.isfinite(f0):
                continue
            f0s.append(f0)
            rep_ids.append(rep)
            for key, est in ffd.items():
                r = run_once(est, f, x0, a.T)
                for m in METRICS + ["f"]:
                    if m != "prog":
                        res[key][m].append(r[m])
            print(f"   rep {rep + 1}/{a.reps}", flush=True)
    finally:
        f.close()
    if not f0s:
        print(f"   [skip] {pname}: no finite starting point", flush=True)
        return

    fmin = get_fmin(pname, a, fstars, [r for k in res for r in res[k]["f"]])
    f0s = np.array(f0s)
    denoms = f0s - fmin
    for key in res:                                              # progress needs f*, so it is computed afterwards
        res[key]["prog"] = [(fr - fmin) / dn if np.isfinite(dn) and dn > 0 else np.full_like(fr, np.nan)
                            for fr, dn in zip(res[key]["f"], denoms)]
    print(f"   f* = {fmin:.6g}", flush=True)

    if raw_w.path:
        for (fam, l), mets in res.items():
            for i, rep in enumerate(rep_ids):
                fr = mets["f"][i]
                raw_w.write([{"function": pname, "family": fam, "dtype": dname, "d": d, "l": l, "rep": rep, "k": k,
                              "f": fr[k], **{m: mets[m][i][k] for m in METRICS}}
                             for k in range(0, len(fr), a.stride)])
    rows = []
    for (fam, l), mets in res.items():
        mats = {m: pad_stack(mets[m]) for m in METRICS}
        K = mats["prog"].shape[1]
        for k in sorted(set(range(0, K, a.stride)) | {K - 1}):
            row = dict(function=pname, family=fam, dtype=dname, d=d, l=l, k=k, h=a.h, fmin=fmin,
                       n_reps=int(np.isfinite(mats["prog"][:, k]).sum()))
            for m in METRICS:
                row.update(summarize(m, mats[m][:, k]))
            rows.append(row)
    summary_w.write(rows)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--problems", nargs="*", default=None, help="CUTEst problem names, e.g. ROSENBR ARGLINA")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--find-unconstrained", action="store_true")
    ap.add_argument("--find-min-n", type=int, default=None)
    ap.add_argument("--find-max-n", type=int, default=None)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--sif-params", nargs="*", default=None, metavar="K=V", help="SIF parameters, e.g. N=100")
    ap.add_argument("--fmin", default="best", help="'best', 'zero' or a JSON file {problem: f_star}")
    ap.add_argument("--families", nargs="+", default=[f for f in FAMILY_ORDER if f in FAMILIES],
                    choices=list(FAMILIES))
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l (default: 2, d/3, d/2, 4d/5, d)")
    ap.add_argument("--dtypes", nargs="+", default=["float64"], choices=list(DTYPES))
    ap.add_argument("--reps", type=int, default=50, help="number of random initialisations")
    ap.add_argument("--T", type=int, default=10000, help="budget of function evaluations per run")
    ap.add_argument("--h", type=float, default=1e-7, help="finite-difference step")
    ap.add_argument("--sample-scale", type=float, default=0.1, help="sigma of the perturbation of x0 (0 = always x0)")
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
    ap.add_argument("--device", default="cpu", help="pycutest runs on CPU; GPU only moves the estimator")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/convergence_ls_cutest.csv")
    ap.add_argument("--raw-out", default=None, help="optional per-repetition, per-iteration CSV (can be large)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    a.sif_params = parse_sif_params(a.sif_params)

    fstars = None
    if a.fmin not in ("best", "zero"):
        with open(a.fmin) as fh:
            fstars = json.load(fh)

    names = select_problems(a)
    if not names:
        ap.error("no problems selected: use --problems or --find-*")

    log_path = a.log or default_log_path(a.out)
    write_log(a, log_path, names)
    print(f"log -> {log_path}; {len(names)} problems", flush=True)

    summary_w, raw_w = Writer(a.out), Writer(a.raw_out)
    t0 = time.time()
    for dname in a.dtypes:
        for pname in names:
            print(f"[++] {pname}", flush=True)
            try:
                run_block(pname, dname, a, fstars, summary_w, raw_w)
                msg = f"[ok] {pname:12s} {dname}  ({time.time() - t0:.0f}s)"
            except Exception as e:
                msg = f"[fail] {pname:12s} {dname}: {type(e).__name__}: {e}"
            print(msg, flush=True)
            append_log(log_path, msg)
    append_log(log_path, f"finished       : {datetime.datetime.now().isoformat(timespec='seconds')} "
                         f"(total {time.time() - t0:.0f}s)")
    print(f"summary -> {a.out}" + (f", raw -> {a.raw_out}" if a.raw_out else ""))


if __name__ == "__main__":
    main()