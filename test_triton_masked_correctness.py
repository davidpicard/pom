#!/usr/bin/env python3
"""Correctness tests for the Triton 1-D mask kernel vs pure PyTorch reference.

Tests the masked (B, N) aggregation path, checking:
  - Forward pass output values
  - Backward pass grad_x and grad_coeff

Tested across a grid of (B, N, D, K) configurations and three mask types:
  - binary (0/1 float)  – typical padding mask
  - dense   (all 1s)    – should agree with the no-mask unmasked mean
  - sparse  (25 % kept) – heavy masking

Usage:
    python test_triton_masked_correctness.py
"""
import sys
import itertools

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    sys.exit("No CUDA device found – skipping correctness tests.")

device = torch.device("cuda")

from pom.pom_triton_masked import poly_agg_masked_triton, TRITON_MASKED_AVAILABLE

if not TRITON_MASKED_AVAILABLE:
    sys.exit("TRITON_MASKED_AVAILABLE=False – nothing to test.")

# ---------------------------------------------------------------------------
# Pure PyTorch reference  (mirrors polynomial_aggregation_ fallback exactly)
# ---------------------------------------------------------------------------

def _pom_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6.0)


def ref_masked(x: torch.Tensor, mask: torch.Tensor,
               coeff: torch.Tensor, k: int) -> torch.Tensor:
    """PyTorch reference: masked weighted mean.  Returns (B, 1, D)."""
    h = _pom_activation(x).unsqueeze(-1)          # (B, N, D, 1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    h = (torch.cat(powers, dim=-1) * coeff).sum(-1)   # (B, N, D)
    m = mask.unsqueeze(-1).to(h.dtype)                 # (B, N, 1)
    return (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_MARK = "\033[32mPASS\033[0m"
FAIL_MARK = "\033[31mFAIL\033[0m"

_TOLS = {
    torch.float32:  dict(atol=1e-4, rtol=1e-3),
    torch.bfloat16: dict(atol=1e-3, rtol=5e-3),
}


def _make_inputs(B, N, D, K, dtype, requires_grad=False):
    torch.manual_seed(42)
    x     = torch.randn(B, N, D, device=device, dtype=dtype) * 0.5
    coeff = torch.randn(D, K, device=device, dtype=dtype) * 0.1
    if requires_grad:
        x     = x.detach().requires_grad_(True)
        coeff = coeff.detach().requires_grad_(True)
    return x, coeff


def _make_mask(B, N, keep_ratio: float, dtype=torch.float32) -> torch.Tensor:
    """Random binary mask with approximately keep_ratio fraction of 1s.

    At least one token per row is always kept to avoid division by zero.
    """
    torch.manual_seed(99)
    m = (torch.rand(B, N, device=device) < keep_ratio).float().to(dtype)
    # Guarantee at least one kept token per row.
    m[:, 0] = 1.0
    return m


def _close(a: torch.Tensor, b: torch.Tensor,
           dtype: torch.dtype) -> tuple[bool, float, float]:
    tols    = _TOLS[dtype]
    a, b    = a.float(), b.float()
    diff    = (a - b).abs()
    thresh  = tols["atol"] + tols["rtol"] * b.abs()
    ok      = bool((diff <= thresh).all())
    max_abs = diff.max().item()
    max_rel = (diff / b.abs().clamp(min=1e-8)).max().item()
    return ok, max_abs, max_rel


# ---------------------------------------------------------------------------
# Forward test
# ---------------------------------------------------------------------------

def test_forward(mask_type: str, keep_ratio: float,
                 Bs, Ns, Ds, Ks, dtype=torch.float32):
    print(f"\n{'='*65}")
    print(f"  forward/masked  mask={mask_type}({keep_ratio:.0%})  dtype={dtype}")
    print(f"{'='*65}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        x, coeff = _make_inputs(B, N, D, K, dtype)
        mask     = _make_mask(B, N, keep_ratio, dtype=dtype)

        with torch.no_grad():
            ref  = ref_masked(x, mask, coeff, K)
            trit = poly_agg_masked_triton(x, mask, coeff, K)

        ok, abs_err, rel_err = _close(ref, trit, dtype)
        if not ok:
            all_pass = False
        status = PASS_MARK if ok else FAIL_MARK
        print(f"  B={B} N={N:4d} D={D:4d} K={K:2d}  "
              f"abs={abs_err:.2e}  rel={rel_err:.2e}  [{status}]")

    return all_pass


# ---------------------------------------------------------------------------
# Backward test
# ---------------------------------------------------------------------------

def test_backward(mask_type: str, keep_ratio: float,
                  Bs, Ns, Ds, Ks, dtype=torch.float32):
    print(f"\n{'='*65}")
    print(f"  backward/masked  mask={mask_type}({keep_ratio:.0%})  dtype={dtype}")
    print(f"{'='*65}")

    all_pass = True
    for B, N, D, K in itertools.product(Bs, Ns, Ds, Ks):
        mask = _make_mask(B, N, keep_ratio, dtype=dtype)

        # --- PyTorch reference backward (always in fp32, matching Triton) ---
        xr_raw, cr_raw = _make_inputs(B, N, D, K, dtype)
        xr = xr_raw.float().detach().requires_grad_(True)
        cr = cr_raw.float().detach().requires_grad_(True)
        mr = mask.float()

        out_ref = ref_masked(xr, mr, cr, K)
        torch.manual_seed(7)
        go_fp32     = torch.randn_like(out_ref)
        go_quantised = go_fp32.to(dtype).float()
        out_ref.backward(go_quantised)
        gx_ref = xr.grad.to(dtype)
        gc_ref = cr.grad.float()

        # --- Triton backward ---
        xt, ct = _make_inputs(B, N, D, K, dtype, requires_grad=True)
        mt = mask.to(dtype)
        out_trit = poly_agg_masked_triton(xt, mt, ct, K)
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
Ks = [1, 2, 3, 4, 8, 16]

Bs_bf = [2]
Ns_bf = [64, 512]
Ds_bf = [64, 128]
Ks_bf = [2, 4]

# Three mask regimes to exercise.
MASKS = [
    ("dense",  1.00),   # all tokens kept  → should match unmasked mean
    ("half",   0.50),   # ~50 % kept
    ("sparse", 0.25),   # ~25 % kept
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []

for mask_name, keep_ratio in MASKS:
    results.append(test_forward(mask_name, keep_ratio, Bs, Ns, Ds, Ks, torch.float32))
    results.append(test_backward(mask_name, keep_ratio, Bs, Ns, Ds, Ks, torch.float32))

# bfloat16 sanity-check (forward + backward) for half-density mask.
results.append(test_forward("half",  0.50, Bs_bf, Ns_bf, Ds_bf, Ks_bf, torch.bfloat16))
results.append(test_backward("half", 0.50, Bs_bf, Ns_bf, Ds_bf, Ks_bf, torch.bfloat16))

print()
print("=" * 65)
if all(results):
    print(f"  {PASS_MARK}  All tests passed.")
else:
    n_fail = results.count(False)
    print(f"  {FAIL_MARK}  {n_fail} test suite(s) failed — see above for details.")
print("=" * 65)
sys.exit(0 if all(results) else 1)
