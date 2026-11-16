import argparse
import datetime
import platform
import sys
import time

import numpy as np
import torch

sys.path.append("../")
sys.path.append("../../")

import pycutest


from exp_utils import (DTYPES, FAMILY_ORDER, STATS, SEED, Writer, append_log, coverage_error, default_log_path,
                       get_ls, summarize, true_grad)
from fdopt.directions.directions import FAMILIES
from fdopt.directions.directions import make as make_operator
from fdopt.optimizers.ffd import FFD
from synthetic_targets import PycutestTarget



def parse_sif_params(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return out


def select_problems(a):
    names = list(a.problems or [])
    if a.find_min_n is not None or a.find_max_n is not None or a.find_unconstrained:
        kw = {}
        if a.find_unconstrained:
            kw["constraints"] = "U"
        if a.find_min_n is not None or a.find_max_n is not None:
            kw["n"] = [a.find_min_n or 1, a.find_max_n or 10 ** 9]
        kw["userN"] = False if (a.find_min_n is not None or a.find_max_n is not None) else None
        kw = {k: v for k, v in kw.items() if v is not None}
        found = pycutest.find_problems(**kw)
        names += [p for p in found if p not in names]
    names = [p for p in names if p not in a.exclude]
    if a.max_problems:
        names = names[: a.max_problems]
    return names


def load_problem(name, sif_params):

    if sif_params:
        try:
            return pycutest.import_problem(name, sifParams=sif_params)
        except Exception as e:  # the problem has no such parameter / invalid value
            print(f"   [warn] {name}: sifParams {sif_params} not accepted ({type(e).__name__}); using default size",
                  flush=True)
    return pycutest.import_problem(name)


def sample_points(prob, f, gen, sigma):

    x0 = f.x0
    if sigma > 0:
        z = torch.randn(1, f.d, generator=gen, dtype=f.dtype, device=f.device)
        x = x0 + sigma * (1 + x0.abs()) * z
    else:
        x = x0.clone()
    bl = getattr(prob, "bl", None)
    bu = getattr(prob, "bu", None)
    if bl is not None and bu is not None:
        lo = torch.from_numpy(np.asarray(bl)).to(dtype=f.dtype, device=f.device).reshape(1, -1)
        hi = torch.from_numpy(np.asarray(bu)).to(dtype=f.dtype, device=f.device).reshape(1, -1)
        x = torch.minimum(torch.maximum(x, lo), hi)
    return x


def write_log(a, path, names):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "Gradient-quality experiment (CUTEst)",
        f"started        : {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"command        : {' '.join(sys.argv)}",
        f"host           : {platform.node()} ({platform.platform()})",
        f"python / torch : {platform.python_version()} / {torch.__version__}",
        f"device         : {a.device}",
        "",
        "Parameters",
        f"problems       : {', '.join(names)}",
        f"sif params     : {a.sif_params}",
        f"families       : {', '.join(a.families)}",
        f"dtypes         : {', '.join(a.dtypes)}",
        f"l              : {a.ls if a.ls else '2, d/3, d/2, 4d/5, d (per problem)'}",
        f"repetitions    : {a.reps}",
        f"h              : {a.h}",
        f"sample-scale   : {a.sample_scale}  (x = x0 + sigma (1+|x0|) N(0,I), clipped to bounds)",
        f"seed           : {a.seed}",
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


def run_block(pname, dname, a, summary_w, raw_w):
    dt, dev = DTYPES[dname], a.device
    prob = load_problem(pname, a.sif_params)
    f = PycutestTarget(prob, dtype=dt, device=dev, seed=a.seed)
    d = f.d
    if d < 2:
        print(f"   [skip] {pname}: d={d}", flush=True)
        f.close()
        return
    ls = get_ls(a, d)
    ffd = {(fam, l): FFD(make_operator(fam, d, l, dt, dev, seed=a.seed), h=a.h) for fam in a.families for l in ls}
    err = {k: [] for k in ffd}
    cov = {k: [] for k in ffd}
    xgen = torch.Generator(device=dev)
    try:
        for rep in range(a.reps):
            xgen.manual_seed(a.seed + 1000 * rep + 7)           # same x for every family and l
            x = sample_points(prob, f, xgen, a.sample_scale)
            gt = true_grad(f, x).to(dt).flatten()
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
                    raw_w.write([{"function": pname, "family": fam, "dtype": dname, "d": d, "l": l, "rep": rep,
                                  "h": a.h, "rel_err": e, "cover_err": ce}])
    finally:
        f.close()
    summary_w.write([dict(function=pname, family=fam, dtype=dname, d=d, l=l, n_reps=len(err[(fam, l)]), h=a.h,
                          **summarize("err", err[(fam, l)]), **summarize("cover", cov[(fam, l)]))
                     for (fam, l) in ffd])


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--problems", nargs="*", default=None, help="CUTEst problem names, e.g. ROSENBR ARGLINA")
    ap.add_argument("--exclude", nargs="*", default=[], help="problem names to skip")
    ap.add_argument("--find-unconstrained", action="store_true", help="add all unconstrained CUTEst problems")
    ap.add_argument("--find-min-n", type=int, default=None, help="add problems with fixed dimension >= this")
    ap.add_argument("--find-max-n", type=int, default=None, help="add problems with fixed dimension <= this")
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--sif-params", nargs="*", default=None, metavar="K=V", help="SIF parameters, e.g. N=100")
    ap.add_argument("--families", nargs="+", default=[f for f in FAMILY_ORDER if f in FAMILIES],
                    choices=list(FAMILIES))
    ap.add_argument("--ls", type=int, nargs="*", default=None, help="values of l (default: 2, d/3, d/2, 4d/5, d)")
    ap.add_argument("--dtypes", nargs="+", default=["float64"], choices=list(DTYPES),
                    help="pycutest works in float64; float32 only casts the values")
    ap.add_argument("--reps", type=int, default=100, help="number of random points")
    ap.add_argument("--h", type=float, default=1e-7, help="finite-difference step")
    ap.add_argument("--sample-scale", type=float, default=0.1,
                    help="sigma of the perturbation of x0 (0 = always x0)")
    ap.add_argument("--device", default="cpu", help="pycutest runs on CPU; GPU only moves the estimator")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/grad_quality_cutest.csv")
    ap.add_argument("--raw-out", default=None, help="optional per-repetition CSV")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    a.sif_params = parse_sif_params(a.sif_params)

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
            try:
                run_block(pname, dname, a, summary_w, raw_w)
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