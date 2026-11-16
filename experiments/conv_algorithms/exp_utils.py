"""Helpers shared by the experiment scripts (extracted from run_grad_acc.py)."""
import csv
import os

import numpy as np
import torch

DTYPES = {"float64": torch.float64, "float32": torch.float32}
FAMILY_ORDER = ["coordinate", "qr", "householder", "permuted_householder", "butterfly",
                "gaussian", "spherical", "rademacher"]
FUNCTIONS = ["LeastSquares", "Qing", "Rosenbrock", "Griewank", "Trid", "RegularizedLogistic"]
STATS = ["mean", "std", "median", "p05", "p25", "p75", "p95", "min", "max"]
SEED = 121212


def l_grid(d):
    """Same grid as the paper: 2, d/3, d/2, 4d/5, d."""
    return sorted({2, max(2, d // 3), max(2, d // 2), max(2, 4 * d // 5), d})


def get_ls(a, d):
    """Values of l actually used for dimension d (`a` is the argparse namespace, uses a.ls)."""
    ls = a.ls if a.ls else l_grid(d)
    return [l for l in ls if 1 <= l <= d]


def sample_point(f, gen, scale):
    """x uniform in the box f.bounds (shrunk around its centre by `scale`), shape (1, d)."""
    lo, hi = f.bounds[:, 0], f.bounds[:, 1]
    centre, half = (lo + hi) / 2, (hi - lo) / 2 * scale
    u = torch.rand(1, f.d, generator=gen, dtype=f.dtype, device=f.device)
    return (centre + half * (2 * u - 1)).reshape(1, -1)


def summarize(prefix, values):
    """Statistics of `values` (non-finite entries dropped), as a dict with keys `<prefix>_<stat>`."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_{s}": float("nan") for s in STATS}
    p05, p25, med, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    return {f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(ddof=1) if v.size > 1 else 0.0,
            f"{prefix}_median": med, f"{prefix}_p05": p05, f"{prefix}_p25": p25, f"{prefix}_p75": p75,
            f"{prefix}_p95": p95, f"{prefix}_min": v.min(), f"{prefix}_max": v.max()}


class Writer:
    """Append rows to a CSV, writing the header once. An existing file is removed at construction."""

    def __init__(self, path):
        self.path, self.fields = path, None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.exists(path):
                os.remove(path)

    def write(self, rows):
        if not self.path or not rows:
            return
        new = self.fields is None
        self.fields = self.fields or list(rows[0])
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerows(rows)


def default_log_path(out):
    return os.path.splitext(out)[0] + ".log"


def append_log(path, text):
    with open(path, "a") as fh:
        fh.write(text + "\n")