import datetime
import gc
import os
import platform
import time

import torch

from results_utils import default_out, write_csv, write_json, stats  # noqa: F401  (re-exported for the experiment scripts)

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
        t = time.time()
        fn()
        t = time.time() - t
        sync(device)
        out.append(t)
    return out


def peak_mem_reps(fn, device, reset=None):
    """Peak extra GPU bytes of ONE call of fn, as a one-element list ([] on CPU, where it is not measured).

    Memory use is deterministic, so a single measurement is enough. `reset` (e.g. op.release) drops the previously
    sampled directions, the cache is emptied, the bytes allocated before the call are noted, the peak counter is
    reset, fn runs, and peak = torch.cuda.max_memory_allocated() - bytes present before. Operator state created in
    the constructor is reported separately by `persistent_bytes`.
    """
    if not is_cuda(device):
        return []
    if reset is not None:
        reset()
    sync(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    res = fn()
    sync(device)
    peak = torch.cuda.max_memory_allocated() - base
    del res
    return [peak]


def persistent_bytes(build, device):
    """(obj, [bytes allocated by build()]); the list is [] on CPU. One measurement (memory is deterministic)."""
    if not is_cuda(device):
        return build(), []
    cleanup(device)
    sync(device)
    base = torch.cuda.memory_allocated()
    obj = build()
    sync(device)
    return obj, [torch.cuda.memory_allocated() - base]


def build_mem(op, device):
    """([peak], [retained]) extra CUDA bytes of ONE op.resample(); ([], []) on CPU.

    peak     = transient maximum while the directions are built
    retained = bytes still held afterwards (explicit: the l x d matrix; implicit: indices / v / seed)"""
    if not is_cuda(device):
        return [], []
    op.release()
    sync(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    op.resample()
    sync(device)
    peak = torch.cuda.max_memory_allocated() - base
    kept = torch.cuda.memory_allocated() - base
    op.release()
    return [peak], [kept]


def build_measured(row, build, a):
    obj, vals = persistent_bytes(build, a.device)
    row.update(stats(vals, "persistent_bytes"))
    return obj


def measure(row, fn, a, reset=None, prefix="time", memory=True):
    times = time_reps(fn, a.device, a.warmup, a.reps)
    row.update(stats(times, prefix))
    row["reps"] = len(times)
    if memory:
        row.update(stats(peak_mem_reps(fn, a.device, reset), "peak_bytes"))
    return times


def record(a, rows, row):
    rows.append(row)
    print(row, flush=True)
    write_csv(a.out, rows)


def l_grid(d):
    return sorted({v for v in (2, d//5, d//3, d//2, int((2/3) * d), int((4/5) * d), d)})


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def add_common(ap, out):
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dims", type=int, nargs="+", default=[500, 5000, 10000])
    ap.add_argument("--dtypes", nargs="+", default=["float64", "float32"], choices=list(DTYPES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20, help="timed repetitions per configuration")
    ap.add_argument("--out", default=out)
    ap.add_argument("--shard", type=int, default=0, help="index of this worker (0 .. nshards-1)")
    ap.add_argument("--nshards", type=int, default=1, help="split the job list over this many processes")


def init_shard(a):
    if a.nshards > 1:
        stem, ext = os.path.splitext(a.out)
        a.out = f"{stem}.shard{a.shard}of{a.nshards}{ext}"
    counter = [0]

    def mine():
        k = counter[0]
        counter[0] += 1
        return k % a.nshards == a.shard
    return mine


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
        memory=("CUDA only, each quantity measured ONCE (memory use is deterministic). persistent_bytes = bytes "
                "allocated by the operator constructor; build_peak_bytes / build_retained_bytes = transient peak / "
                "bytes kept by one resample() (explicit: building the l x d matrix; implicit: indices, v, seed); "
                "peak_bytes = peak extra bytes (torch.cuda.max_memory_allocated minus bytes present before) of one "
                "call of the measured function (implicit: computing the action; estimator experiments: the whole "
                "estimator); footprint = peak + persistent. Columns keep the *_mean/_median/... names with a "
                "single sample. NaN on CPU."),
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