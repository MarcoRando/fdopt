import argparse
import csv
import math
import os
import sys

import torch
from bench import (
    DTYPES,
    add_common,
    build_measured,
    build_mem,
    cleanup,
    finish,
    init_shard,
    is_oom,
    l_grid,
    measure,
    randn_x,
    record,
)
from results_utils import default_out, make_row, read_csv, stats

sys.path.append("../../")

from fdopt.directions.directions import FAMILIES, make  # noqa: E402


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


def batch_sizes(l, max_iters):
    cand = {1, l // 3, l // 2, int((2 / 3) * l), l}
    return sorted({b for b in cand if b > 0})


def job_key(r):
    """Parameters that identify one run."""
    return (r["method"], r["family"], r["part"], r["dtype"], int(r["d"]), int(r["l"]), int(r["batch"]))


def load_done(path, retry_oom):
    """Rows already stored in `path` (kept, so the file is extended rather than rewritten) and their keys.

    Stops if the stored columns differ from the current schema (mixing them would corrupt the csv)."""
    if not os.path.exists(path):
        return [], set()
    with open(path, newline="") as fh:
        header = next(csv.reader(fh), [])
    expected = list(make_row("exp2", method="implicit", family="x", part="ask", dtype="float64", d=1, l=1, batch=1))
    if header != expected:
        raise SystemExit(f"{path} has different columns than the current exp2 schema; use --overwrite or another --out")
    rows = read_csv(path)
    if retry_oom:
        rows = [r for r in rows if r.get("status") == "ok"]
    return rows, {job_key(r) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    add_common(ap, default_out("exp2_action.csv"))
    ap.add_argument("--parts", nargs="+", default=["both"], choices=["ask", "tell", "both"])
    ap.add_argument("--max-iters", type=int, default=20000, help="skip batch sizes needing more chunks than this")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES),
                    help="direction families of fdopt.directions.directions")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing result file and start from scratch")
    ap.add_argument("--retry-oom", action="store_true", help="re-run runs stored with status 'oom'")
    a = ap.parse_args()
    mine = init_shard(a)
    h = 1e-6
    # runs already present in the result file are kept and skipped (every shard reads its own file)
    rows, done = ([], set()) if a.overwrite else load_done(a.out, a.retry_oom)
    if done:
        print(f"[resume] {len(done)} runs already in {a.out} will be skipped", flush=True)
    skipped = 0
    for dname in a.dtypes:
        dt = DTYPES[dname]
        for d in a.dims:
            x = randn_x(d, dt, a.device)
            for l in l_grid(d):
                s = torch.randn(l, dtype=dt, device=a.device, generator=torch.Generator(device=a.device).manual_seed(1))
                batches = batch_sizes(l, a.max_iters)
                for part in a.parts:
                    jobs = [("implicit", n, b) for n in a.families for b in batches]   # library operators only
                    jobs = [j for j in jobs if mine()]      # every shard enumerates all jobs and keeps its own
                    for method, name, b in jobs:
                        if (method, name, part, dname, d, l, b) in done:     # already measured: skip
                            skipped += 1
                            continue
                        row = make_row("exp2", method=method, family=name, part=part, dtype=dname, d=d, l=l, batch=b)
                        op = fn = None
                        try:
                            op = build_measured(row, lambda: make(name, d, l, dt, a.device), a)
                            # memory of building the directions (resample)
                            peaks, kept = build_mem(op, a.device)          # lists of one value ([] on CPU)
                            row.update(stats(peaks, "build_peak_bytes"))
                            row.update(stats(kept, "build_retained_bytes"))
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
    if skipped:
        print(f"[resume] skipped {skipped} runs already done", flush=True)
    finish(a, rows)


if __name__ == "__main__":
    main()
