"""Exp 1 -- direction generation only (explicit full matrix), per dtype.

Answers: how expensive is the Haar/QR matrix in float32 vs float64, and does re-orthogonalising a float32
matrix (and casting to double) fix the orthogonality loss?  `qr_reorth` = one extra QR pass in the working dtype,
`qr_f32_reorth` = QR in float32, a second QR pass in float32, cast to float64 (only run for dtype float64).
"""
import argparse

import torch
import sys
from bench import DTYPES, add_common, cleanup, finish, is_oom, l_grid, measure, peak_mem, persistent_bytes, record
from results_utils import make_row

sys.path.append("../../")

from fdopt.directions.directions import FAMILIES, VARIANTS, make


def main():
    ap = argparse.ArgumentParser()
    add_common(ap, "results/exp1_generation.csv")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES) + list(VARIANTS))
    a = ap.parse_args()
    rows = []
    for dname in a.dtypes:
        dt = DTYPES[dname]
        for d in a.dims:
            for l in l_grid(d):
                for name in a.families:
                    if name == "qr_f32_reorth" and dname != "float64":
                        continue
                    row = make_row("exp1", method="explicit", family=name, dtype=dname, d=d, l=l)
                    try:
                        op, row["persistent_bytes"] = persistent_bytes(lambda: make(name, d, l, dt, a.device), a.device)
                        measure(row, op.generate, a, memory=False)
                        row["peak_bytes"] = peak_mem(op.generate, a.device)
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