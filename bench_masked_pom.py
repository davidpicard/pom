#!/usr/bin/env python3
"""Benchmark: PyTorch masked PoM vs Triton masked PoM.

The 1-D mask (B, N) path aggregates only selected tokens.  The PyTorch
fallback materialises a full (B, N, D, K) intermediate before masking;
the Triton kernel fuses activation + polynomial + masked sum in one pass.

Three configurations are compared at each sequence length N:
  pytorch-masked   PyTorch fallback (pom_activation → poly expansion → mask_mixer)
  triton-masked    New Triton kernel (single fused pass)
  triton-unmasked  No-mask Triton kernel (poly_agg_mean_triton) — speed ceiling

Masks with ~50 % of tokens kept are used throughout.

Both forward and backward passes are timed.  N is swept from 64 to 8192.

Usage:
    python bench_masked_pom.py
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping benchmark.")

device = torch.device("cuda")

from pom.pom_triton         import poly_agg_mean_triton
from pom.pom_triton_masked  import poly_agg_masked_triton

print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print()

# ---------------------------------------------------------------------------
# Reference PyTorch path (replicates polynomial_aggregation_ fallback exactly,
# bypassing the Triton dispatch so we always get the pure-PyTorch numbers).
# ---------------------------------------------------------------------------

def _pom_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6.0)


def pytorch_masked(x: torch.Tensor, mask: torch.Tensor,
                   coeff: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-PyTorch masked aggregation — no Triton."""
    h = _pom_activation(x).unsqueeze(-1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    h = (torch.cat(powers, dim=-1) * coeff).sum(-1)
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Wrappers with identical signatures  (x, mask, coeff, k) → (B, 1, D)
# ---------------------------------------------------------------------------

def run_pytorch_masked(x, mask, coeff, k):
    return pytorch_masked(x, mask, coeff, k)

def run_triton_masked(x, mask, coeff, k):
    return poly_agg_masked_triton(x, mask, coeff, k)

def run_triton_unmasked(x, mask, coeff, k):
    # mask is ignored — measures upper-bound speed without mask overhead
    return poly_agg_mean_triton(x, coeff, k)


RUNNERS = {
    "pytorch-masked" : run_pytorch_masked,
    "triton-masked"  : run_triton_masked,
    "triton-unmasked": run_triton_unmasked,
}

# ---------------------------------------------------------------------------
# Helpers
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


def time_fwd(runner, x, mask, coeff, k):
    return median_time(lambda: runner(x, mask, coeff, k))


def time_bwd(runner, x, mask, coeff, k):
    def _step():
        if x.grad is not None:   x.grad = None
        if coeff.grad is not None: coeff.grad = None
        return runner(x, mask, coeff, k).sum()

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

DIM    = 1024
K      = 4          # polynomial degree
DTYPE  = torch.bfloat16
B      = 64
KEEP   = 0.50       # fraction of tokens kept in the mask

Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

COLS = list(RUNNERS.keys())

def print_header():
    col_w = 16
    hdr = f"{'N':>6}  " + "  ".join(f"{c:>{col_w}}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))

def print_row(N, times: dict):
    col_w = 16
    vals  = []
    speedup = None
    if times.get("pytorch-masked") and times.get("triton-masked"):
        speedup = times["pytorch-masked"] / times["triton-masked"]
    for name in COLS:
        t = times.get(name)
        vals.append(f"{'OOM':>{col_w}}" if t is None else f"{t:>{col_w}.3f}")
    suffix = f"  (triton speedup: {speedup:.2f}×)" if speedup else ""
    print(f"{N:>6}  " + "  ".join(vals) + suffix)


coeff_param = torch.randn(DIM, K, device=device, dtype=DTYPE)

print(f"dim={DIM}, K={K}, B={B}, dtype={DTYPE}, keep={KEEP:.0%}")
print()

for section, time_fn in [("FORWARD", time_fwd), ("BACKWARD", time_bwd)]:
    req_grad = (section == "BACKWARD")

    print("=" * 80)
    print(f"{section}  (B={B}, dim={DIM}, K={K}, bfloat16, mask={KEEP:.0%} kept, "
          f"median 100 runs, ms)")
    print("=" * 80)
    print()
    print_header()

    for N in Ns:
        torch.manual_seed(0)
        mask = (torch.rand(B, N, device=device) < KEEP).float().to(DTYPE)
        mask[:, 0] = 1.0   # guarantee at least one kept token

        coeff = coeff_param.detach().requires_grad_(req_grad)
        times = {}

        for name, runner in RUNNERS.items():
            x = torch.randn(B, N, DIM, device=device, dtype=DTYPE,
                            requires_grad=req_grad)
            t = try_run(lambda r=runner, x=x, m=mask, c=coeff, k=K:
                        time_fn(r, x, m, c, k))
            times[name] = t

        print_row(N, times)

    print()
