import datetime
import gc
import os
import platform
import time

import torch

from results_utils import write_csv, write_json, stats  # noqa: F401  (re-exported for the experiment scripts)

DTYPES = {"float64": torch.float64, "float32": torch.float32}

TIMING_DESCRIPTION = (
    "wall clock (time.perf_counter) of ONE call of the measured function per repetition; on CUDA, "
    "torch.cuda.synchronize() is called before the timer starts and after the call returns, so asynchronous "
    "kernels are fully counted and nothing leaks between repetitions")


def is_cuda(device):
    return str(device).startswith("cuda")


def sync(device):
    if is_cuda(device):
        torch.cuda.synchronize()


def cleanup(device):
    gc.collect()
    if is_cuda(device):
        torch.cuda.empty_cache()


def time_reps(fn, device, warmup=3, reps=20):
    """Per-repetition wall-clock times (seconds): `warmup` untimed calls, then `reps` individually timed calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(reps):
        sync(device)
        t = time.perf_counter()
        fn()
        sync(device)
        out.append(time.perf_counter() - t)
    return out


def peak_mem(fn, device, reset=None):
    """Peak extra GPU bytes allocated by one call of fn, relative to the allocations present before the call.

    `reset` (e.g. op.release) is called first so previously sampled directions are not counted as present.
    Operator state created in the constructor is reported separately by `persistent_bytes`.
    Returns -1 on CPU.
    """
    if not is_cuda(device):
        return -1
    if reset is not None:
        reset()
    sync(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    sync(device)
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def persistent_bytes(build, device):
    """(build(), bytes allocated by build()); bytes = -1 on CPU."""
    if not is_cuda(device):
        return build(), -1
    sync(device)
    base = torch.cuda.memory_allocated()
    obj = build()
    sync(device)
    return obj, torch.cuda.memory_allocated() - base


def measure(row, fn, a, reset=None, prefix="time", memory=True):
    """Time fn over a.reps repetitions (after a.warmup), store median/percentile columns (+ peak memory) in row."""
    times = time_reps(fn, a.device, a.warmup, a.reps)
    row.update(stats(times, prefix))
    row["reps"] = len(times)
    if memory:
        row["peak_bytes"] = peak_mem(fn, a.device, reset)
    return times


def record(a, rows, row):
    """Append a row, echo it, and checkpoint the CSV so a crash on a long GPU run loses nothing."""
    rows.append(row)
    print(row, flush=True)
    write_csv(a.out, rows)


def l_grid(d):
    return sorted({v for v in (1, 10, 100, 1000, 10000, d // 2, d) if 1 <= v <= d})


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def add_common(ap, out):
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dims", type=int, nargs="+", default=[1000, 4000, 10000])
    ap.add_argument("--dtypes", nargs="+", default=["float64", "float32"], choices=list(DTYPES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20, help="timed repetitions per configuration")
    ap.add_argument("--out", default=out)


def randn_x(d, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1, d, dtype=dtype, device=device, generator=g)


def f_quadratic(X):
    return 0.5 * X.square().sum(dim=1)


def make_f(r):
    """Objective whose cost grows with r (r tanh sweeps over the batch); r = 0 is the plain quadratic."""
    def f(X):
        Z = X.clone()
        for _ in range(r):
            Z.tanh_()
        return 0.5 * Z.square().sum(dim=1)
    return f


def env_info(a=None, extra=None):
    """Everything the reviewers asked to be reported about the measurement setup."""
    info = dict(
        implementation_language="Python",
        python=platform.python_version(),
        torch=torch.__version__,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        torch_num_threads=torch.get_num_threads(),
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        timing=TIMING_DESCRIPTION,
        statistic="median and 5/25/75/95th percentiles over the timed repetitions (stored per configuration)",
        allocations_included=("yes: allocating/generating the directions (resample) is inside the timed region of "
                              "'ask', 'both', the generation and the estimator/iteration experiments; "
                              "'tell' times accumulation only, with directions generated beforehand"),
        memory=("peak_bytes = peak extra CUDA bytes of one call (torch.cuda.max_memory_allocated minus bytes present "
                "before the call, previously sampled directions released); persistent_bytes = bytes allocated by "
                "the operator constructor; footprint = peak_bytes + persistent_bytes"),
        seeds="operators seeded with 121212, evaluation point with seed 0; the operator RNG advances across repetitions",
    )
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, gpu_total_memory_bytes=p.total_memory, cuda=torch.version.cuda)
    if a is not None:
        info["args"] = {k: v for k, v in vars(a).items()}
        info["device"] = str(getattr(a, "device", ""))
    if extra:
        info.update(extra)
    return info


def finish(a, rows, extra=None):
    write_csv(a.out, rows)
    write_json(os.path.splitext(a.out)[0] + ".meta.json", env_info(a, extra))