#!/usr/bin/env python3
"""Benchmark: PyTorch RoPE path vs fused Triton RoPE kernels.

Compares the three-step PyTorch chain
    poly_features → apply_rope_1d → aggregate
against the single-pass Triton kernel for:
  - unmasked  (global mean)
  - causal    (causal prefix mean)

Both forward and backward are timed.  N is swept 64 → 8192.

Usage:
    python bench_rope_triton.py
"""
import sys
import time

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping benchmark.")

device = torch.device("cuda")

from pom.pom_rope import (
    precompute_freqs_1d,
    apply_rope_1d,
    _poly_features,
    _aggregate,
)
from pom.pom_triton_rope import (
    poly_agg_rope_mean_triton,
    poly_agg_rope_causal_triton,
    TRITON_ROPE_AVAILABLE,
)

if not TRITON_ROPE_AVAILABLE:
    sys.exit("TRITON_ROPE_AVAILABLE=False – skipping benchmark.")

print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print()

# ---------------------------------------------------------------------------
# PyTorch reference runners
# ---------------------------------------------------------------------------

def pytorch_mean(x, coeff, k, fc, fs, pos):
    h = _poly_features(x, coeff, k)
    h = apply_rope_1d(h, pos, fc, fs)
    return _aggregate(h, None)

def pytorch_causal(x, coeff, k, fc, fs, pos):
    h = _poly_features(x, coeff, k)
    h = apply_rope_1d(h, pos, fc, fs)
    return _aggregate(h, "causal")

def triton_mean(x, coeff, k, fc, fs, pos):
    return poly_agg_rope_mean_triton(x, coeff, k, fc, fs, pos)

def triton_causal(x, coeff, k, fc, fs, pos):
    return poly_agg_rope_causal_triton(x, coeff, k, fc, fs, pos)

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def median_time(fn, warmup=10, reps=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def time_fwd(runner, x, coeff, k, fc, fs, pos):
    return median_time(lambda: runner(x, coeff, k, fc, fs, pos))


def time_bwd(runner, x, coeff, k, fc, fs, pos):
    def _step():
        if x.grad is not None:     x.grad = None
        if coeff.grad is not None: coeff.grad = None
        return runner(x, coeff, k, fc, fs, pos).sum()

    for _ in range(10):
        _step().backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        loss = _step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def try_run(fn):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIM   = 1024
K     = 4
DTYPE = torch.bfloat16
B     = 64

Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

RUNNERS = {
    "pytorch-mean"   : (pytorch_mean,   "mean"),
    "triton-mean"    : (triton_mean,    "mean"),
    "pytorch-causal" : (pytorch_causal, "causal"),
    "triton-causal"  : (triton_causal,  "causal"),
}

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

COL_W = 18

def print_header(names):
    hdr = f"{'N':>6}  " + "  ".join(f"{n:>{COL_W}}" for n in names)
    print(hdr)
    print("-" * len(hdr))


def print_row(N, times, col_names):
    vals = []
    for name in col_names:
        t = times.get(name)
        vals.append(f"{'OOM':>{COL_W}}" if t is None else f"{t:>{COL_W}.3f}")
    # speedup pairs
    suffixes = []
    for pt, tr in [("pytorch-mean", "triton-mean"), ("pytorch-causal", "triton-causal")]:
        if times.get(pt) and times.get(tr):
            suffixes.append(f"{pt.split('-')[1]} ×{times[pt]/times[tr]:.2f}")
    suffix = "  (" + ", ".join(suffixes) + ")" if suffixes else ""
    print(f"{N:>6}  " + "  ".join(vals) + suffix)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

coeff_param = torch.randn(DIM, K, device=device, dtype=DTYPE)
max_N = max(Ns)
fc_full, fs_full = precompute_freqs_1d(DIM, max_N + 16)
fc_full = fc_full.to(device=device, dtype=DTYPE)
fs_full = fs_full.to(device=device, dtype=DTYPE)

COLS = list(RUNNERS.keys())

print(f"dim={DIM}, K={K}, B={B}, dtype={DTYPE}")
print()

for section, time_fn in [("FORWARD", time_fwd), ("BACKWARD", time_bwd)]:
    req_grad = (section == "BACKWARD")

    print("=" * 90)
    print(f"{section}  (B={B}, dim={DIM}, K={K}, bfloat16, median 100 runs, ms)")
    print("=" * 90)
    print()
    print_header(COLS)

    for N in Ns:
        torch.manual_seed(0)
        pos   = torch.arange(N, device=device, dtype=torch.int64)
        fc    = fc_full[:N]
        fs    = fs_full[:N]
        coeff = coeff_param.detach().requires_grad_(req_grad)
        times = {}

        for name, (runner, _) in RUNNERS.items():
            x = torch.randn(B, N, DIM, device=device, dtype=DTYPE,
                            requires_grad=req_grad)
            t = try_run(lambda r=runner, x=x, c=coeff, k=K, fc=fc, fs=fs, p=pos:
                        time_fn(r, x, c, k, fc, fs, p))
            times[name] = t

        print_row(N, times, COLS)

    print()
