"""Exp 2 -- generating (ask) and applying (tell) the directions, explicit vs implicit, per batch size.

part = ask  : resample + build the perturbed points x + h P[a:e] for all chunks
part = tell : g += s @ P[a:e] for all chunks (directions prepared beforehand)
part = both : ask + tell (what one estimator costs, without evaluating f)

Memory columns (CUDA only, scalars, each measured once):
  persistent_bytes       operator constructor (identity matrix, cached constants, ...)
  build_peak_bytes       transient peak of resample()   -> explicit: building the l x d matrix
  build_retained_bytes   bytes kept after resample()    -> explicit: the matrix; implicit: indices / v
  peak_bytes             transient peak of the measured part -> implicit: computing the action (ask/tell/both)
Parallel use: --shard i --nshards n (one process per GPU, see parallel.py).
"""
import argparse
import math

import torch

from bench import (DTYPES, add_common, build_mem, build_measured, cleanup, finish, init_shard, is_oom, l_grid,
                   measure, randn_x, record)

import sys

sys.path.append("../../")

from fdopt.directions.directions import FAMILIES, make
from results_utils import default_out, make_row


def factory(op, x, s, h, b, part):
    l = op.l

    def build():
        if part == "tell":
            op.resample()
            op.perturb(x, h, 0, min(b, l))

        def fn():
            if part != "tell":
                op.resample()
            g = torch.zeros_like(x)
            for a in range(0, l, b):
                e = min(a + b, l)
                if part in ("ask", "both"):
                    op.perturb(x, h, a, e)
                if part in ("tell", "both"):
                    op.accumulate(g, s[a:e], a, e)
            if part in ("tell", "both"):
                op.finalize(g)
            return g
        return fn
    return build


def main():
    ap = argparse.ArgumentParser()
    add_common(ap, default_out("exp2_action.csv"))
#    ap.add_argument("--batches", type=int, nargs="+", default=[1, 10, 100, 500])
    ap.add_argument("--parts", nargs="+", default=["ask", "tell", "both"], choices=["ask", "tell", "both"])
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--modes", nargs="+", default=["implicit", "explicit"], choices=["implicit", "explicit"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    a = ap.parse_args()
    mine = init_shard(a)
    h = 1e-6
    rows = []
    for dname in a.dtypes:
        dt = DTYPES[dname]
        for d in a.dims:
            x = randn_x(d, dt, a.device)
            for l in l_grid(d):
                s = torch.randn(l, dtype=dt, device=a.device, generator=torch.Generator(device=a.device).manual_seed(1))
                batches = sorted({b for b in [1, l//3, l//2, int((2/3) * l), l]})# if l > 10 and math.ceil(l / b) <= a.max_iters} | {l})
                for part in a.parts:
                    jobs = []
                    if "implicit" in a.modes:
                        jobs += [("implicit", n, b) for n in a.families if n in IMPLICIT_FAMILIES for b in batches]
                    if "explicit" in a.modes:
                        jobs += [("explicit", n, l) for n in a.families]
                    jobs = [j for j in jobs if mine()]      # every shard enumerates all jobs and keeps its own
                    for method, name, b in jobs:
                        row = make_row("exp2", method=method, family=name, part=part, dtype=dname, d=d, l=l, batch=b)
                        try:
                            op = build_measured(
                                row, lambda: make(name, d, l, dt, a.device, implicit=(method == "implicit")), a)
                            # memory of building the directions (explicit: the l x d matrix)
                            row["build_peak_bytes"], row["build_retained_bytes"] = build_mem(op, a.device)
                            fn = factory(op, x, s, h, b, part)()
                            # for 'tell' the sampled directions are part of the prepared state, so they are not released
                            measure(row, fn, a, reset=None if part == "tell" else op.release)
                        except Exception as ex:
                            if not is_oom(ex):
                                raise
                            row["status"] = "oom"
                        fn = op = None
                        cleanup(a.device)
                        record(a, rows, row)
    finish(a, rows)


if __name__ == "__main__":
    main()