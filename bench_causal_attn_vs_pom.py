#!/usr/bin/env python3
"""Benchmark: Causal Flash Attention vs Triton-optimized Causal PoM.

Three mixers are compared, all with the same (B, N, dim) input and output:

  flash-causal   F.scaled_dot_product_attention with is_causal=True.
  pom-causal-k{2-5}  PoM with mask="causal", using the Triton causal kernel.
                     degree 2 to 5 (comparable param count to attention).

Both forward and backward passes are timed. N is swept from 64 to 8192 to
expose the O(N²) vs O(N) scaling difference.

Usage:
    python bench_causal_attn_vs_pom.py
"""
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping benchmark.")

device = torch.device("cuda")

from pom import PoM

print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print()


# =============================================================================
# Mixer modules  (all take (B, N, D) → (B, N, D), causal, eval mode)
# =============================================================================

class CausalFlashAttnMixer(nn.Module):
    """Causal attention using F.scaled_dot_product_attention (is_causal=True)."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CausalPoMMixer(nn.Module):
    """PoM with causal masking (mask="causal" → Triton causal kernel on CUDA).

    n_sel_heads=num_heads matches the attention param count comparison.
    """
    def __init__(self, dim: int, num_heads: int, expand: int = 1, degree: int = 3):
        super().__init__()
        self.pom = PoM(dim=dim, degree=degree, expand=expand,
                       n_groups=1, n_sel_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pom(x, mask="causal")


# =============================================================================
# Helpers
# =============================================================================

def param_count(module: nn.Module) -> str:
    n = sum(p.numel() for p in module.parameters())
    return f"{n / 1e6:.2f}M"


def median_time(fn, warmup: int = 10, reps: int = 100) -> float:
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


def time_fwd(module, x):
    return median_time(lambda: module(x))


def time_bwd(module, x):
    """Times backward only (forward is excluded)."""
    def _step():
        if x.grad is not None:
            x.grad = None
        for p in module.parameters():
            if p.grad is not None:
                p.grad = None
        return module(x).sum()

    # Warm up (includes forward+backward)
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
    """Returns None on OOM, cleans up cache."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


# =============================================================================
# Configuration
# =============================================================================

DIM       = 1024
NUM_HEADS = 16
DTYPE     = torch.bfloat16
B         = 4

Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


# =============================================================================
# Instantiate modules
# =============================================================================

mixers = {
    "flash-causal" : CausalFlashAttnMixer(DIM, NUM_HEADS),
    "pom-causal-k2": CausalPoMMixer(DIM, num_heads=NUM_HEADS, expand=2, degree=2),
    "pom-causal-k3": CausalPoMMixer(DIM, num_heads=NUM_HEADS, expand=2, degree=3),
    "pom-causal-k4": CausalPoMMixer(DIM, num_heads=NUM_HEADS, expand=2, degree=4),
    "pom-causal-k5": CausalPoMMixer(DIM, num_heads=NUM_HEADS, expand=2, degree=5),
}

for name, m in mixers.items():
    m.to(device=device, dtype=DTYPE).eval()

print(f"dim={DIM}, num_heads={NUM_HEADS}, B={B}, dtype={DTYPE}")
print()
print(f"  {'mixer':<16}  params")
print(f"  {'-'*16}  ------")
for name, m in mixers.items():
    print(f"  {name:<16}  {param_count(m)}")
print()


# =============================================================================
# Benchmark loop
# =============================================================================

COLS = list(mixers.keys())

def print_header():
    col_w = 14
    hdr = f"{'N':>6}  " + "  ".join(f"{c:>{col_w}}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))

def print_row(N, times: dict):
    col_w = 14
    vals = []
    for name in COLS:
        t = times.get(name)
        if t is None:
            vals.append(f"{'OOM':>{col_w}}")
        else:
            vals.append(f"{t:>{col_w}.3f}")
    print(f"{N:>6}  " + "  ".join(vals))


for section, time_fn in [("FORWARD", time_fwd), ("BACKWARD", time_bwd)]:
    req_grad = (section == "BACKWARD")

    print("=" * 80)
    print(f"{section}  (B={B}, dim={DIM}, bfloat16, median 100 runs, ms)")
    print("=" * 80)
    print()
    print_header()

    for N in Ns:
        times = {}
        for name, m in mixers.items():
            x = torch.randn(B, N, DIM, device=device, dtype=DTYPE,
                            requires_grad=req_grad)
            for p in m.parameters():
                p.requires_grad_(req_grad)

            t = try_run(lambda m=m, x=x: time_fn(m, x))
            times[name] = t

        print_row(N, times)

    print()
