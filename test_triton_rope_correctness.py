#!/usr/bin/env python3
"""Correctness tests for the fused Triton RoPE kernels vs pure PyTorch reference.

Tests poly_agg_rope_mean_triton  (no-mask)  and
      poly_agg_rope_causal_triton (causal mean)

for both forward and backward passes, over a grid of (B, N, D, K) configs,
default positions (0..N-1), and float32 / bfloat16.

Usage:
    python test_triton_rope_correctness.py
"""
import sys
import itertools

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping.")

device = torch.device("cuda")

from pom.pom_triton_rope import (
    poly_agg_rope_mean_triton,
    poly_agg_rope_causal_triton,
    TRITON_ROPE_AVAILABLE,
)
from pom.pom_rope import precompute_freqs_1d

if not TRITON_ROPE_AVAILABLE:
    sys.exit("TRITON_ROPE_AVAILABLE=False – nothing to test.")

# ---------------------------------------------------------------------------
# Pure PyTorch references  (always fp32 internally, matching Triton's policy)
# ---------------------------------------------------------------------------

def _pom_act(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6.0)


def _poly_features_fp32(x: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """Polynomial expansion, fp32 throughout."""
    x = x.float()
    coeff = coeff.float()
    h = _pom_act(x).unsqueeze(-1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    return (torch.cat(powers, dim=-1) * coeff).sum(-1)   # (B, N, D)


def _apply_rope_fp32(h: torch.Tensor, positions: torch.Tensor,
                     fc: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
    """Rotate-half in fp32."""
    D    = h.shape[-1]
    half = D // 2
    cos  = fc[positions].float()   # (N, D//2)
    sin  = fs[positions].float()
    h1, h2 = h[..., :half], h[..., half:]
    return torch.cat([h1 * cos - h2 * sin, h2 * cos + h1 * sin], dim=-1)


def ref_rope_mean(x, coeff, k, fc, fs, positions):
    """(B,1,D) — global mean after RoPE, always fp32."""
    h = _poly_features_fp32(x, coeff, k)
    h = _apply_rope_fp32(h, positions, fc, fs)
    return h.mean(dim=1, keepdim=True)


def ref_rope_causal(x, coeff, k, fc, fs, positions):
    """(B,N,D) — causal prefix mean after RoPE, always fp32."""
    h   = _poly_features_fp32(x, coeff, k)
    h   = _apply_rope_fp32(h, positions, fc, fs)
    B, N, D = h.shape
    acc = torch.zeros(B, D, device=h.device, dtype=torch.float32)
    out = torch.empty(B, N, D, device=h.device, dtype=torch.float32)
    for n in range(N):
        acc = acc + h[:, n, :]
        out[:, n, :] = acc / (n + 1)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_MARK = "\033[32mPASS\033[0m"
FAIL_MARK = "\033[31mFAIL\033[0m"

_TOLS = {
    torch.float32:  dict(atol=1e-4, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}


def _close(a, b, dtype):
    tols   = _TOLS[dtype]
    a, b   = a.float(), b.float()
    diff   = (a - b).abs()
    thresh = tols["atol"] + tols["rtol"] * b.abs()
    ok     = bool((diff <= thresh).all())
    return ok, diff.max().item(), (diff / b.abs().clamp(min=1e-8)).max().item()


def _make_inputs(B, N, D, K, dtype, requires_grad=False, seed=42):
    torch.manual_seed(seed)
    x     = torch.randn(B, N, D, device=device, dtype=dtype) * 0.5
    coeff = torch.randn(D, K, device=device, dtype=dtype) * 0.1
    if requires_grad:
        x     = x.detach().requires_grad_(True)
        coeff = coeff.detach().requires_grad_(True)
    return x, coeff


def _make_freqs(D, N, dtype=torch.float32):
    fc, fs = precompute_freqs_1d(D, N + 16)   # slight overallocation
    return fc.to(device=device, dtype=dtype), fs.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

def test_forward(variant, ref_fn, triton_fn, Bs, Ns, Ds, Ks, dtype=torch.float32):
    print(f"\n{'='*65}")
    print(f"  forward/{variant}  dtype={dtype}")
    print(f"{'='*65}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        x, coeff = _make_inputs(B, N, D, K, dtype)
        fc, fs   = _make_freqs(D, N)
        pos      = torch.arange(N, device=device, dtype=torch.int64)

        with torch.no_grad():
            ref  = ref_fn(x, coeff, K, fc, fs, pos)
            trit = triton_fn(x, coeff, K, fc, fs, pos)

        ok, abs_e, rel_e = _close(ref, trit, dtype)
        if not ok:
            all_pass = False
        mark = PASS_MARK if ok else FAIL_MARK
        print(f"  B={B} N={N:4d} D={D:4d} K={K:2d}  "
              f"abs={abs_e:.2e}  rel={rel_e:.2e}  [{mark}]")

    return all_pass


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

def test_backward(variant, ref_fn, triton_fn, Bs, Ns, Ds, Ks, dtype=torch.float32):
    print(f"\n{'='*65}")
    print(f"  backward/{variant}  dtype={dtype}")
    print(f"{'='*65}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        fc, fs = _make_freqs(D, N)
        pos    = torch.arange(N, device=device, dtype=torch.int64)

        # --- Reference: fp32 inputs, go round-tripped through dtype ---
        xr_raw, cr_raw = _make_inputs(B, N, D, K, dtype)
        xr = xr_raw.float().detach().requires_grad_(True)
        cr = cr_raw.float().detach().requires_grad_(True)

        out_ref = ref_fn(xr, cr, K, fc, fs, pos)
        torch.manual_seed(7)
        go_fp32     = torch.randn_like(out_ref)
        go_quantised = go_fp32.to(dtype).float()
        out_ref.backward(go_quantised)
        gx_ref = xr.grad.to(dtype)
        gc_ref = cr.grad.float()

        # --- Triton ---
        xt, ct = _make_inputs(B, N, D, K, dtype, requires_grad=True)
        out_trit = triton_fn(xt, ct, K, fc, fs, pos)
        out_trit.backward(go_fp32.to(dtype))
        gx_trit = xt.grad.clone()
        gc_trit = ct.grad.float()

        ok_x, abs_x, rel_x = _close(gx_ref, gx_trit, dtype)
        ok_c, abs_c, rel_c = _close(gc_ref, gc_trit, dtype)
        ok = ok_x and ok_c
        if not ok:
            all_pass = False

        sx = PASS_MARK if ok_x else FAIL_MARK
        sc = PASS_MARK if ok_c else FAIL_MARK
        print(f"  B={B} N={N:4d} D={D:4d} K={K:2d}  "
              f"grad_x abs={abs_x:.2e} rel={rel_x:.2e} [{sx}]  "
              f"grad_c abs={abs_c:.2e} rel={rel_c:.2e} [{sc}]")

    return all_pass


# ---------------------------------------------------------------------------
# Test grid
# ---------------------------------------------------------------------------

Bs = [1, 4]
Ns = [16, 64, 256, 1024]
Ds = [32, 64, 96, 128, 512]   # 96 is non-power-of-2
Ks = [1, 2, 4, 8, 16]

Bs_bf = [2]
Ns_bf = [64, 512]
Ds_bf = [64, 128]
Ks_bf = [2, 4]

results = []

for variant, ref_fn, triton_fn in [
    ("mean",   ref_rope_mean,   poly_agg_rope_mean_triton),
    ("causal", ref_rope_causal, poly_agg_rope_causal_triton),
]:
    results.append(test_forward(variant,  ref_fn, triton_fn, Bs, Ns, Ds, Ks, torch.float32))
    results.append(test_backward(variant, ref_fn, triton_fn, Bs, Ns, Ds, Ks, torch.float32))

# bfloat16 forward + backward sanity check
for variant, ref_fn, triton_fn in [
    ("mean",   ref_rope_mean,   poly_agg_rope_mean_triton),
    ("causal", ref_rope_causal, poly_agg_rope_causal_triton),
]:
    results.append(test_forward(variant,  ref_fn, triton_fn, Bs_bf, Ns_bf, Ds_bf, Ks_bf, torch.bfloat16))
    results.append(test_backward(variant, ref_fn, triton_fn, Bs_bf, Ns_bf, Ds_bf, Ks_bf, torch.bfloat16))

print()
print("=" * 65)
if all(results):
    print(f"  {PASS_MARK}  All tests passed.")
else:
    n_fail = results.count(False)
    print(f"  {FAIL_MARK}  {n_fail} test suite(s) failed — see above.")
print("=" * 65)
sys.exit(0 if all(results) else 1)
