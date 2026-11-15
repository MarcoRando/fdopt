import argparse

import torch

from bench import DTYPES, add_common, build_measured, cleanup, finish, init_shard, is_oom, l_grid, measure, record
from exp_utils import EXPLICIT_FAMILIES, make_explicit
from results_utils import default_out, make_row


def main():
    ap = argparse.ArgumentParser()
    add_common(ap, default_out("exp1_generation.csv"))
    ap.add_argument("--families", nargs="+", default=EXPLICIT_FAMILIES)
    a = ap.parse_args()
    mine = init_shard(a)
    rows = []
    for dname in a.dtypes:
        dt = DTYPES[dname]
        for d in a.dims:
            for l in l_grid(d):
                for name in a.families:
                    if name == "qr_f32_reorth" and dname != "float64":
                        continue
                    if not mine():
                        continue
                    row = make_row("exp1", method="explicit", family=name, dtype=dname, d=d, l=l)
                    try:
                        op = build_measured(row, lambda: make_explicit(name, d, l, dt, a.device), a)
                        measure(row, op.generate, a)
                        if op.orthogonal and l * d <= 10 ** 8:
                            P = op.generate().double()
                            row["orth_err"] = (P @ P.T - torch.eye(l, dtype=torch.float64, device=P.device)).abs().max().item()
                            del P
                    except Exception as ex:
                        if not is_oom(ex):
                            raise
                        row["status"] = "oom"
                    op = None
                    cleanup(a.device)
                    record(a, rows, row)
    finish(a, rows)


if __name__ == "__main__":
    main()