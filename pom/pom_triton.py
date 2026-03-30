"""Triton-accelerated kernels for PoM (Polynomial Mixer).

The hot path in polynomial_aggregation_ (no-mask branch) is:
  1. pom_activation  (element-wise)
  2. polynomial weighted sum  producing (B, N, D)
  3. mean over N  producing (B, 1, D)

PyTorch materialises a (B, N, D, K) intermediate before the mean.
The kernels here fuse all three steps into a single pass over the
input, eliminating the intermediate and halving memory bandwidth.

Forward kernel optimisations
-----------------------------
- coeff[d, k] is preloaded once per (b, d_blk) program and held in
  registers for the entire N loop.
- X is streamed with eviction_policy="evict_first" to avoid polluting L2.
- cache_modifier=".ca" on coeff loads provides an L1 fallback on spill.
- K is tl.constexpr: the polynomial loop is fully unrolled.
- BLOCK_D is chosen by a Python heuristic (no autotuning, no file I/O).

Backward kernel optimisations
-------------------------------
- A single fused kernel computes both grad_x and grad_coeff in one pass
  over X, so X is read only once (vs. twice with separate kernels).
- go (upstream gradient) and coeff are preloaded into registers.
- The hp accumulator is shared: the same power vector serves both the
  grad_h computation (for grad_x) and the coeff accumulator.
- grad_coeff is accumulated per-batch-element and written to the output
  buffer via tl.atomic_add, avoiding a separate reduction pass.
- grad_x is stored as fp32; cast to input dtype in Python wrapper.

Exposed API
-----------
TRITON_AVAILABLE : bool
poly_agg_mean_triton(x, coeff, k) -> Tensor  (B, 1, D)
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    # ---------------------------------------------------------------------------
    # BLOCK_D heuristics — computed in Python, no autotuning, no file I/O.
    #
    # Forward: up to 256 (streaming kernel, light register pressure).
    # Backward: up to 128 (heavier: coeff + grad_coeff accumulators in regs).
    # Both: next power of 2 >= D, clamped to the respective cap.
    # ---------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # ---------------------------------------------------------------------------
    # Forward kernel
    # ---------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_mean_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, D)    – output, fp32
        N,              # sequence length (runtime)
        D,              # feature dim    (runtime)
        stride_xb,      # X.stride(0) = N*D
        stride_xn,      # X.stride(1) = D
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] once per program; held in registers.
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

        tl.store(O_ptr + b * D + d_off, acc / N, mask=dmask)

    # ---------------------------------------------------------------------------
    # Fused backward kernel – grad_x and grad_coeff in one X pass
    #
    # Grid: (B, ceil(D/BLOCK_D))
    #   Each program handles one batch element and one D tile.
    #   After the N loop it atomic_adds its partial grad_coeff sums into GC,
    #   reducing across B without a separate reduction kernel.
    #
    # Power sharing:
    #   hp advances as h^0 → h^1 → ... → h^(K-1).
    #   At step k:
    #     grad_h update uses hp  (= h^k)           → coeff derivative
    #     acc[k]  update uses hp * h (= h^(k+1))   → coeff gradient
    #   The same hp * h product serves both, so no extra multiply.
    # ---------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_mean_bwd(
        GO_ptr,         # (B, D)    – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # --- Preload: constant over the N loop ---
        go    = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)
        inv_N = 1.0 / N

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

        # --- grad_coeff accumulators (one per degree, reduced across N) ---
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

        for n in range(N):
            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # hp advances as h^0, h^1, ..., h^(K-1).
            # At step k: grad_h uses hp (= h^k); a[k] uses hp*h (= h^(k+1)).
            # hph = hp * h is computed once and serves both, then hp is advanced.
            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;   a0 += go * hph;   hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1 += go * hph;   hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2 += go * hph;   hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3 += go * hph;   hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4 += go * hph;   hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5 += go * hph;   hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6 += go * hph;   hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7 += go * hph;   hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8 += go * hph;   hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9 += go * hph;   hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += go * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += go * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += go * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += go * hph;  hp = hph
            if K > 14:
                hph = hp * h
                grad_h += c14 * 15.0 * hp; a14 += go * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += go * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                go * inv_N * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        # GC is zero-initialised by the Python wrapper before this kernel runs.
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0  * inv_N, mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1  * inv_N, mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2  * inv_N, mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3  * inv_N, mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4  * inv_N, mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5  * inv_N, mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6  * inv_N, mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7  * inv_N, mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8  * inv_N, mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9  * inv_N, mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10 * inv_N, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11 * inv_N, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12 * inv_N, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13 * inv_N, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14 * inv_N, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15 * inv_N, mask=dmask)

    # ---------------------------------------------------------------------------
    # autograd.Function wrapper
    # ---------------------------------------------------------------------------

    class _PolyAggMean(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            # Do NOT call x.contiguous().  The kernel uses stride_xb / stride_xn,
            # so it correctly reads a non-contiguous slice such as
            # qc_proj_out[..., :po_dim] without an extra copy.
            coeff_c = coeff.float().contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_mean_fwd[grid](
                x, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, coeff_c)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, coeff = ctx.saved_tensors
            k              = ctx.k
            # Make contiguous for backward: the bwd kernel uses the same pointer
            # layout for both reading X and writing GX, so a single stride set
            # is needed.  The extra copy here is acceptable (bwd is not the
            # bottleneck this optimisation targets).
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            # fp32 buffers; cast grad_x to input dtype after the kernel.
            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            # GC must be zero-initialised: the kernel accumulates via atomic_add.
            grad_c = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_mean_bwd[grid](
                go, x, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None

    # ---------------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------------

    def poly_agg_mean_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused polynomial aggregation + mean (no mask).

        Args:
            x     : (B, N, D) input tensor
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, 1, D) mean-aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )
        return _PolyAggMean.apply(x, coeff, k)
