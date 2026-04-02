#!/usr/bin/env python3
"""Correctness tests for Triton PoM kernels vs pure PyTorch reference.

Tests both the unmasked (full aggregation) and causal paths, checking:
  - Forward pass output values
  - Backward pass grad_x and grad_coeff

Each test iterates over a grid of (B, N, D, K) configurations and reports
the maximum absolute error between the Triton and PyTorch outputs.  A test
is marked PASS when the error is below the threshold, FAIL otherwise.

Usage:
    python test_triton_correctness.py
"""
import sys
import itertools

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping correctness tests.")

device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Import the Triton entry points directly so we can call PyTorch and Triton
# side-by-side without going through polynomial_aggregation_'s dispatch logic.
# ---------------------------------------------------------------------------
from pom.pom_triton import poly_agg_mean_triton, TRITON_AVAILABLE
from pom.pom_triton_causal import poly_agg_causal_triton, TRITON_CAUSAL_AVAILABLE

if not TRITON_AVAILABLE:
    sys.exit("TRITON_AVAILABLE=False – nothing to test.")
if not TRITON_CAUSAL_AVAILABLE:
    sys.exit("TRITON_CAUSAL_AVAILABLE=False – nothing to test.")

# ---------------------------------------------------------------------------
# Pure PyTorch reference implementations
# (mirrors the fallback branches in polynomial_aggregation_)
# ---------------------------------------------------------------------------

def _pom_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6.0)


def _poly_eval(x: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """(B, N, D) → (B, N, D) after polynomial expansion + weighted sum."""
    h = _pom_activation(x).unsqueeze(-1)          # (B, N, D, 1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    return (torch.cat(powers, dim=-1) * coeff).sum(-1)   # (B, N, D)


def ref_unmasked(x: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """PyTorch reference for the no-mask (full mean) path.  Returns (B, 1, D)."""
    return _poly_eval(x, coeff, k).mean(dim=1, keepdim=True)


def ref_causal(x: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """PyTorch reference for the causal path.  Returns (B, N, D)."""
    h = _poly_eval(x, coeff, k)                   # (B, N, D)
    B, N, D = h.shape
    # Build lower-triangular weight matrix once on GPU; (N, N)
    tril = torch.tril(torch.ones(N, N, device=h.device, dtype=h.dtype))
    # (B, N, D): out[b, m, d] = sum_{n<=m} h[b,n,d] / (m+1)
    out = torch.einsum("bnd,mn->bmd", h, tril)
    counts = tril.sum(dim=1).view(1, N, 1)        # (1, N, 1) = 1,2,...,N
    return out / counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_MARK = "\033[32mPASS\033[0m"
FAIL_MARK = "\033[31mFAIL\033[0m"

# Tolerances follow torch.allclose semantics: |a-b| <= atol + rtol * |b|.
# fp32 accumulation with high polynomial degree (K=16 → h^16, h up to ~1.5
# with x*0.5 scaling) makes pure absolute tolerance useless; a relative
# component is essential.
#
# bfloat16 note: the Triton kernels compute entirely in fp32 (inputs are
# upcast) but the grad_coeff reduction uses tl.atomic_add whose evaluation
# order across batch elements differs from PyTorch's parallel reduction.
# For bfloat16, this causes ~0.35 % relative error on grad_coeff — well
# within the inherent 1-ULP precision of bfloat16 (~0.4 %).  A per-dtype
# relative tolerance is therefore appropriate.
_TOLS = {
    torch.float32:  dict(atol=1e-4, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-3, rtol=5e-3),
}


def _make_inputs(B, N, D, K, dtype, requires_grad=False):
    """Create reproducible random (x, coeff) on CUDA.

    x is scaled to std ≈ 0.5 so that pom_activation(x) typically stays in
    (-0.1, 1.5).  This keeps h^K small enough for K=16 to remain within fp32
    precision under both the sequential Triton loop and PyTorch's autograd.
    """
    torch.manual_seed(42)
    x     = torch.randn(B, N, D, device=device, dtype=dtype) * 0.5
    coeff = torch.randn(D, K, device=device, dtype=dtype) * 0.1
    if requires_grad:
        x     = x.detach().requires_grad_(True)
        coeff = coeff.detach().requires_grad_(True)
    return x, coeff


def _close(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> tuple[bool, float, float]:
    """Return (ok, max_abs_err, max_rel_err) using mixed atol+rtol criterion."""
    tols   = _TOLS[dtype]
    a, b   = a.float(), b.float()
    diff   = (a - b).abs()
    thresh = tols["atol"] + tols["rtol"] * b.abs()
    ok     = bool((diff <= thresh).all())
    max_abs = diff.max().item()
    denom   = b.abs().clamp(min=1e-8)
    max_rel = (diff / denom).max().item()
    return ok, max_abs, max_rel


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

def test_forward(path, Bs, Ns, Ds, Ks, dtype=torch.float32):
    assert path in ("unmasked", "causal")
    ref_fn    = ref_unmasked   if path == "unmasked" else ref_causal
    triton_fn = poly_agg_mean_triton if path == "unmasked" else poly_agg_causal_triton

    label = f"forward/{path}"
    print(f"\n{'='*60}")
    print(f"  {label}  dtype={dtype}")
    print(f"{'='*60}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        x, coeff = _make_inputs(B, N, D, K, dtype)
        with torch.no_grad():
            ref  = ref_fn(x, coeff, K)
            trit = triton_fn(x, coeff, K)
        ok, abs_err, rel_err = _close(ref, trit, dtype)
        if not ok:
            all_pass = False
        status = PASS_MARK if ok else FAIL_MARK
        print(f"  B={B} N={N:4d} D={D:4d} K={K:2d}  "
              f"abs={abs_err:.2e}  rel={rel_err:.2e}  [{status}]")

    return all_pass


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

def test_backward(path, Bs, Ns, Ds, Ks, dtype=torch.float32):
    assert path in ("unmasked", "causal")
    ref_fn    = ref_unmasked   if path == "unmasked" else ref_causal
    triton_fn = poly_agg_mean_triton if path == "unmasked" else poly_agg_causal_triton

    label = f"backward/{path}"
    print(f"\n{'='*60}")
    print(f"  {label}  dtype={dtype}")
    print(f"{'='*60}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        # --- reference backward ---
        # The Triton kernels:
        #   1. Read x (input dtype) and upcast to fp32.
        #   2. Read go (output dtype) and upcast to fp32.
        #   3. Compute everything in fp32.
        #   4. Return grad_x cast back to input dtype; grad_coeff stays fp32.
        # The reference must replicate this exactly to get a fair comparison.
        xr_raw, cr_raw = _make_inputs(B, N, D, K, dtype)

        # fp32 versions of the quantised inputs (identical to what Triton reads)
        xr = xr_raw.float().detach().requires_grad_(True)
        cr = cr_raw.float().detach().requires_grad_(True)
        out_ref = ref_fn(xr, cr, K)

        torch.manual_seed(7)
        # go must pass through the output dtype (round-trip quantisation) so
        # that Triton and the reference see the same numerical go values.
        go_fp32 = torch.randn_like(out_ref)
        go_quantised = go_fp32.to(dtype).float()   # mimic Triton's load+upcast
        out_ref.backward(go_quantised)
        gx_ref = xr.grad.to(dtype)    # cast to match Triton's output dtype
        gc_ref = cr.grad.float()

        # --- Triton backward ---
        xt, ct = _make_inputs(B, N, D, K, dtype, requires_grad=True)
        out_trit = triton_fn(xt, ct, K)
        # Feed go in the output dtype; the kernel will upcast it to fp32.
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

# Cover power-of-2, non-power-of-2, small and large D.
Bs  = [1, 4]
Ns  = [16, 64, 256, 1024]
Ds  = [32, 64, 96, 128, 512]   # 96 is non-power-of-2
Ks  = [1, 2, 3, 4, 8, 16]

# Narrower grid for bfloat16 (lower precision — just sanity-check a few combos).
Bs_bf  = [2]
Ns_bf  = [64, 512]
Ds_bf  = [64, 128]
Ks_bf  = [2, 4]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []

results.append(test_forward("unmasked", Bs, Ns, Ds, Ks, dtype=torch.float32))
results.append(test_forward("causal",   Bs, Ns, Ds, Ks, dtype=torch.float32))
results.append(test_forward("unmasked", Bs_bf, Ns_bf, Ds_bf, Ks_bf, dtype=torch.bfloat16))
results.append(test_forward("causal",   Bs_bf, Ns_bf, Ds_bf, Ks_bf, dtype=torch.bfloat16))

results.append(test_backward("unmasked", Bs, Ns, Ds, Ks, dtype=torch.float32))
results.append(test_backward("causal",   Bs, Ns, Ds, Ks, dtype=torch.float32))
results.append(test_backward("unmasked", Bs_bf, Ns_bf, Ds_bf, Ks_bf, dtype=torch.bfloat16))
results.append(test_backward("causal",   Bs_bf, Ns_bf, Ds_bf, Ks_bf, dtype=torch.bfloat16))

print()
print("=" * 60)
if all(results):
    print(f"  {PASS_MARK}  All tests passed.")
else:
    n_fail = results.count(False)
    print(f"  {FAIL_MARK}  {n_fail} test suite(s) failed — see above for details.")
print("=" * 60)
sys.exit(0 if all(results) else 1)
