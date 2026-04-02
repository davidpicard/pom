"""Triton-accelerated causal-mask kernel for PoM (Polynomial Mixer).

For the causal mask, output position m aggregates context positions 0..m:

  out[b, m, d] = (1/(m+1)) * sum_{n=0}^{m} poly(act(x[b, n, d]))

where poly(h) = sum_{k=0}^{K-1} coeff[d, k] * h^(k+1).

This is O(N) in compute and memory (same as the no-mask case) but produces
(B, N, D) instead of (B, 1, D), enabling autoregressive sequence mixing.

Forward kernel
--------------
Grid: (B, ceil(D/BLOCK_D))
Each program streams n = 0..N-1, accumulates a running polynomial sum `acc`,
and writes `acc / (n+1)` to out[b, n, :] at each step.
Coefficients are preloaded into registers once per program; X is streamed
with evict_first to avoid L2 pollution.

Backward kernel
---------------
Grid: (B, ceil(D/BLOCK_D))
Gradient of the loss w.r.t. x[b, n0, d]:
  grad_x[b, n0, d] = suffix_w[b, n0, d] * d_poly/d_h * d_act/d_x

where suffix_w[b, n, d] = sum_{m=n}^{N-1} go[b, m, d] / (m+1) is the
suffix-weighted sum of upstream gradients.

The kernel iterates n = N-1 .. 0 (via i = 0..N-1, n = N-1-i):
  - Accumulates suffix_w in reverse (one load of go per step).
  - Loads x[b, n, d], computes activation, d_act, grad_h.
  - Writes grad_x[b, n, d] = suffix_w * grad_h * d_act.
  - Accumulates partial grad_coeff sums a[k] += suffix_w * h^(k+1).
After the loop, atomic_add flushes a[k] into GC (reducing across B).

Exposed API
-----------
TRITON_CAUSAL_AVAILABLE : bool
poly_agg_causal_triton(x, coeff, k) -> Tensor  (B, N, D)
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_CAUSAL_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_CAUSAL_AVAILABLE = False


if TRITON_CAUSAL_AVAILABLE:

    # -------------------------------------------------------------------------
    # BLOCK_D heuristics – same policy as the no-mask kernel.
    # -------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # -------------------------------------------------------------------------
    # Forward kernel
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_causal_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, N, D) – output, fp32
        N,              # sequence length  (runtime)
        D,              # feature dim      (runtime)
        stride_xb,      # X.stride(0)
        stride_xn,      # X.stride(1)
        stride_ob,      # O.stride(0)
        stride_on,      # O.stride(1)
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] into registers for the entire N loop.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation: clamp(leaky_relu(x, 0.01), -0.1, 6.0)
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp
            if K > 2:
                hp *= h; poly += c2 * hp
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += poly
            # Causal mean: divide running sum by (n+1) — number of tokens seen so far.
            tl.store(
                O_ptr + b * stride_ob + n * stride_on + d_off,
                acc / (n + 1),
                mask=dmask,
            )

    # -------------------------------------------------------------------------
    # Backward kernel
    #
    # Iterates n = N-1 .. 0 (via i = 0 .. N-1, n = N-1-i) to accumulate the
    # suffix-weighted upstream gradient:
    #   suffix_w[n] = sum_{m=n}^{N-1} go[b, m, d] / (m+1)
    #
    # At each step:
    #   grad_x[b, n, d] = suffix_w[n] * grad_h(h[n]) * d_act(x[n])
    #   a[k]           += suffix_w[n] * h[n]^(k+1)   (→ grad_coeff[d, k])
    #
    # Power sharing: same hp / hph trick as the no-mask backward.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_causal_bwd(
        GO_ptr,         # (B, N, D) – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        stride_gob, stride_gon,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coefficients (constant over the N loop).
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        # grad_coeff accumulators (reduced across N, then atomic-added across B).
        a0  = tl.zeros((BLOCK_D,), tl.float32)
        a1  = tl.zeros((BLOCK_D,), tl.float32)
        a2  = tl.zeros((BLOCK_D,), tl.float32)
        a3  = tl.zeros((BLOCK_D,), tl.float32)
        a4  = tl.zeros((BLOCK_D,), tl.float32)
        a5  = tl.zeros((BLOCK_D,), tl.float32)
        a6  = tl.zeros((BLOCK_D,), tl.float32)
        a7  = tl.zeros((BLOCK_D,), tl.float32)
        a8  = tl.zeros((BLOCK_D,), tl.float32)
        a9  = tl.zeros((BLOCK_D,), tl.float32)
        a10 = tl.zeros((BLOCK_D,), tl.float32)
        a11 = tl.zeros((BLOCK_D,), tl.float32)
        a12 = tl.zeros((BLOCK_D,), tl.float32)
        a13 = tl.zeros((BLOCK_D,), tl.float32)
        a14 = tl.zeros((BLOCK_D,), tl.float32)
        a15 = tl.zeros((BLOCK_D,), tl.float32)

        # Running suffix-weighted upstream gradient (accumulated in reverse).
        suffix_w = tl.zeros((BLOCK_D,), tl.float32)

        for i in range(N):
            n = N - 1 - i   # reverse: n goes N-1, N-2, ..., 0

            # Accumulate suffix_w += go[b, n, d] / (n+1)
            go_n = tl.load(
                GO_ptr + b * stride_gob + n * stride_gon + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            suffix_w += go_n / (n + 1)

            # Load x[b, n, d] for activation / derivative computation.
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            # Derivative of pom_activation:
            #   1    for  0 <= x <= 6
            #   0.01 for -10 <= x < 0
            #   0    otherwise (clamped)
            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # Power-sharing loop – hp advances h^0 → h^1 → ... → h^(K-1).
            # At step k:
            #   grad_h  uses hp (= h^k)           → polynomial derivative
            #   a[k]    uses hph (= h^(k+1))       → coeff gradient
            # The same hph product serves both.
            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;          a0  += suffix_w * hph;  hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1  += suffix_w * hph;  hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2  += suffix_w * hph;  hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3  += suffix_w * hph;  hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4  += suffix_w * hph;  hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5  += suffix_w * hph;  hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6  += suffix_w * hph;  hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7  += suffix_w * hph;  hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8  += suffix_w * hph;  hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9  += suffix_w * hph;  hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += suffix_w * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += suffix_w * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += suffix_w * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += suffix_w * hph;  hp = hph
            if K > 14:
                hph = hp * h
                grad_h += c14 * 15.0 * hp; a14 += suffix_w * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += suffix_w * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                suffix_w * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        # GC is zero-initialised by the Python wrapper before this kernel runs.
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0,  mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1,  mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2,  mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3,  mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4,  mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5,  mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6,  mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7,  mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8,  mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9,  mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15, mask=dmask)

    # -------------------------------------------------------------------------
    # autograd.Function wrapper
    # -------------------------------------------------------------------------

    class _PolyAggCausal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            coeff_c = coeff.float().contiguous()
            out     = torch.empty(B, N, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_causal_fwd[grid](
                x, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                out.stride(0), out.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, coeff_c)
            ctx.k = k
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, coeff = ctx.saved_tensors
            k              = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.float().contiguous()  # (B, N, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            # GC must be zero-initialised: kernel accumulates via atomic_add.
            grad_c = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_causal_bwd[grid](
                go, x, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                go.stride(0), go.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def poly_agg_causal_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused causal polynomial aggregation.

        For each position m, computes the polynomial mean over tokens 0..m:
          out[b, m, d] = (1/(m+1)) * sum_{n=0}^{m} poly(act(x[b, n, d]))

        Args:
            x     : (B, N, D) input tensor
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, N, D) causal-mean-aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton causal kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )
        return _PolyAggCausal.apply(x, coeff, k)
