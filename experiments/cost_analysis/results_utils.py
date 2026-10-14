"""Torch-free helpers: repetition statistics, CSV schemas, CSV I/O, best-config selection.

Kept free of torch so that the plotting scripts can run on any machine.
"""
import csv
import json
import math
import os

NAN = float("nan")
STAT_KEYS = ("median", "p05", "p25", "p75", "p95", "min", "max")
_Q = {"p05": 0.05, "p25": 0.25, "median": 0.5, "p75": 0.75, "p95": 0.95}


def percentile(values, q):
    """Linear-interpolation percentile (same convention as numpy's default), q in [0, 1]."""
    v = sorted(values)
    n = len(v)
    if n == 0:
        return NAN
    if n == 1:
        return v[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def stats(values, prefix="time", keys=STAT_KEYS):
    """Summary of repeated measurements as flat columns: <prefix>_median, _p05, _p25, ..."""
    out = {}
    for k in keys:
        if k == "min":
            out[f"{prefix}_{k}"] = min(values) if values else NAN
        elif k == "max":
            out[f"{prefix}_{k}"] = max(values) if values else NAN
        else:
            out[f"{prefix}_{k}"] = percentile(values, _Q[k])
    return out


def _s(prefix, keys=STAT_KEYS):
    return {f"{prefix}_{k}": NAN for k in keys}


_IQR = ("median", "p25", "p75")

# None  -> field is required in make_row; any other value is the default.
SCHEMAS = {
    "exp1": {"method": None, "family": None, "dtype": None, "d": None, "l": None, **_s("time"),
             "reps": 0, "peak_bytes": -1, "persistent_bytes": -1, "orth_err": NAN, "status": "ok"},
    "exp2": {"method": None, "family": None, "part": None, "dtype": None, "d": None, "l": None, "batch": None,
             **_s("time"), "reps": 0, "peak_bytes": -1, "persistent_bytes": -1, "status": "ok"},
    "exp3": {"method": None, "family": None, "dtype": None, "d": None, "l": None, "batch": None, **_s("time"),
             "reps": 0, "peak_bytes": -1, "persistent_bytes": -1, "rel_err": NAN, "status": "ok"},
    "exp4": {"method": None, "family": None, "dtype": None, "d": None, "l": None, "batch": None, "cost": None,
             **_s("t_est"), **_s("t_f", _IQR), **_s("t_dir", _IQR), "dir_frac": NAN, "reps": 0, "status": "ok"},
    "exp5": {"method": None, "family": None, "dtype": None, "budget_gb": None, "batch": None, "max_d": 0},
    "exp6": {"family": None, "cache": None, "part": None, "dtype": None, "d": None, "l": None, "batch": None,
             **_s("time"), "reps": 0, "peak_bytes": -1, "persistent_bytes": -1, "status": "ok"},
    "exp7": {"method": None, "family": None, "dtype": None, "d": None, "l": None, "batch": None, "cost": None,
             "iters": None, **_s("time"), "reps": 0, "peak_bytes": -1, "f_ratio": NAN, "linesearch": "none",
             "n_evals": NAN, "status": "ok"},
}


def make_row(schema, **fields):
    """Build a row with all columns of `schema`; unknown or missing-required fields fail immediately."""
    base = SCHEMAS[schema]
    unknown = set(fields) - set(base)
    if unknown:
        raise KeyError(f"{schema}: unknown columns {sorted(unknown)}")
    missing = [k for k, v in base.items() if v is None and k not in fields]
    if missing:
        raise KeyError(f"{schema}: missing required columns {missing}")
    row = dict(base)
    row.update(fields)
    return row


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _conv(v):
    if v in ("True", "False"):
        return v == "True"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def read_csv(path):
    """Read a result CSV with numbers/bools converted; [] if the file does not exist."""
    try:
        with open(path) as fh:
            return [{k: _conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    except FileNotFoundError:
        return []


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def footprint(r):
    """Persistent operator bytes + transient peak bytes of a row (-1 if memory was not measured, e.g. CPU)."""
    if r.get("peak_bytes", -1) < 0:
        return -1
    return r["peak_bytes"] + max(r.get("persistent_bytes", 0), 0)


def select_best(rows, dtype, d, l, slack=4.0, family=None, method="implicit", part="both"):
    """Fastest (median time) config whose memory is within `slack` x the smallest one.

    Returns (family, batch) or None. Memory filter is skipped when memory was not measured.
    """
    cand = [r for r in rows if r.get("method") == method and r.get("part") == part and r["dtype"] == dtype
            and r["d"] == d and r["l"] == l and r["status"] == "ok" and math.isfinite(r["time_median"])
            and (family is None or r["family"] == family)]
    if not cand:
        return None
    mem = [footprint(r) for r in cand]
    mmin = min(mem)
    ok = cand if mmin <= 0 else [r for r, m in zip(cand, mem) if m <= slack * mmin]
    r = min(ok, key=lambda r: r["time_median"])
    return r["family"], int(r["batch"])