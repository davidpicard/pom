"""Rotary Position Embedding (RoPE) for the Polynomial Mixer (PoM).

Mathematical justification
--------------------------
PoM computes:

    y_i = S(X_i) ⊙ (1/|M| · Σ_j M_j · poly(act(W_c X_j)))

where S is a gating signal and poly(act(·)) is the polynomial nonlinearity.

With RoPE, each token j's polynomial features are rotated by R_j before
aggregation, and the query gating signal is rotated by R_i before the
element-wise product:

    H̃_j = R_j · poly(act(W_c X_j))          (per-context-token)
    S̃_i = R_i · gate(W_s X_i)               (per-query-token)
    y_i  = S̃_i ⊙ (1/|M| · Σ_j M_j · H̃_j)

**Why this placement is correct.**  For the rotate-half convention the
contribution of context token j to query position i in the d-th / (d+D/2)-th
dimension pair sums to:

    (S̃_i)^d  (H̃_j)^d + (S̃_i)^{d+D/2} (H̃_j)^{d+D/2}
        = (s^d h^d + s^{d+D/2} h^{d+D/2}) cos((j−i)θ)
        + (s^d h^{d+D/2} − s^{d+D/2} h^d) sin((j−i)θ)

which depends **only on the relative position (j−i)** — the same property
that makes RoPE work for dot-product attention.  The final linear projection
ag_proj learns to combine paired dimensions, recovering this dependence.

Applying the rotation *before* the nonlinearity would mix all frequency
components inside the polynomial, destroying the rotation structure.
Applying it *after* aggregation would give no per-token position signal.
The correct placement is: after poly(act(·)), before aggregation / gating.

Note on n_sel_heads > 1
-----------------------
When n_sel_heads > 1 the selection signal S has shape (B, T, n_sel_heads) —
one scalar per head.  A scalar rotation cannot reproduce the exact relative-
position identity, but it still injects absolute position into the gating
signal.  Frequency alignment: the k-th dimension pair of S is assigned the
same base frequency as head k of H, obtained by striding the precomputed
frequency table at step head_dim // 2 (1-D) or head_dim (2-D).  This ensures
S^k oscillates at the same rate as the dominant frequency of H^{k,:}, giving
the best possible approximate alignment within the dimensional constraint.

Requires n_sel_heads even for 1-D RoPE; divisible by 4 for 2-D RoPE.

Supported mask types
--------------------
  mask = None           full mean → (B, 1, D)
  mask = "causal"       causal mean (lower-triangular) → (B, N, D)
  mask : (B, N)         1-D padding mask → (B, 1, D)
  mask : (B, M, N)      cross-attention mask → (B, M, D)

Position conventions
--------------------
  1-D  positions : (N,) int64 — token indices 0..N-1
  2-D  positions_hw : (N, 2) int64 — (row, col) patch coordinates

For self-mixing (xq is xc) a single set of positions applies to both the
context (H) and the query (S).  For cross-attention the caller may supply
separate ctx_positions / ctx_positions_hw for the context side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .pom import (
    PoM,
    pom_activation,
    polynomial_selection_,
    mask_mixer,
    full_mask_mixer,
)

try:
    from .pom_triton_rope import (
        poly_agg_rope_mean_triton,
        poly_agg_rope_causal_triton,
        TRITON_ROPE_AVAILABLE,
    )
except ImportError:
    TRITON_ROPE_AVAILABLE = False


# =============================================================================
# Stand-alone RoPE utilities
# =============================================================================

def precompute_freqs_1d(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency tables for 1-D RoPE.

    Uses the standard schedule θ_k = 1 / base^(2k / dim) for k = 0…dim/2−1.
    The rotate-half convention pairs dimension k with dimension k + dim/2.

    Args:
        dim         : feature dimension (must be even)
        max_seq_len : maximum sequence length
        base        : frequency base (default 10 000)

    Returns:
        freqs_cos : (max_seq_len, dim // 2)
        freqs_sin : (max_seq_len, dim // 2)
    """
    assert dim % 2 == 0, f"dim must be even for 1-D RoPE, got {dim}"
    half     = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t        = torch.arange(max_seq_len, dtype=torch.float32)
    angles   = torch.outer(t, inv_freq)          # (max_seq_len, dim // 2)
    return angles.cos(), angles.sin()


def precompute_freqs_2d(
    dim: int,
    max_h: int,
    max_w: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for 2-D RoPE.

    The feature vector is split into two halves:
      • first  dim/2 dimensions → row position (via 1-D RoPE with dim/4 bands)
      • second dim/2 dimensions → column position (via 1-D RoPE with dim/4 bands)

    Args:
        dim         : feature dimension (must be divisible by 4)
        max_h       : maximum grid height
        max_w       : maximum grid width
        base        : frequency base (default 10 000)

    Returns:
        row_cos, row_sin : (max_h, dim // 4)
        col_cos, col_sin : (max_w, dim // 4)
    """
    assert dim % 4 == 0, f"dim must be divisible by 4 for 2-D RoPE, got {dim}"
    quarter  = dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
    rows     = torch.arange(max_h, dtype=torch.float32)
    cols     = torch.arange(max_w, dtype=torch.float32)
    row_ang  = torch.outer(rows, inv_freq)        # (max_h, dim // 4)
    col_ang  = torch.outer(cols, inv_freq)        # (max_w, dim // 4)
    return row_ang.cos(), row_ang.sin(), col_ang.cos(), col_ang.sin()


def apply_rope_1d(
    x: torch.Tensor,
    positions: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 1-D rotary position embedding to x.

    Uses the rotate-half convention: dimension k is paired with k + D/2.

    Args:
        x         : (B, N, D)
        positions : (N,) int64 position indices
        freqs_cos : (max_len, D // 2) — precomputed cosines
        freqs_sin : (max_len, D // 2) — precomputed sines

    Returns:
        (B, N, D) rotated tensor
    """
    D    = x.shape[-1]
    half = D // 2
    cos  = freqs_cos[positions]      # (N, D // 2)
    sin  = freqs_sin[positions]      # (N, D // 2)
    x1   = x[..., :half]             # (B, N, D // 2)
    x2   = x[..., half:]             # (B, N, D // 2)
    return torch.cat([
        x1 * cos - x2 * sin,         # rotated first  half
        x2 * cos + x1 * sin,         # rotated second half
    ], dim=-1)


def apply_rope_2d(
    x: torch.Tensor,
    positions_hw: torch.Tensor,
    row_cos: torch.Tensor,
    row_sin: torch.Tensor,
    col_cos: torch.Tensor,
    col_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 2-D rotary position embedding to x.

    The feature vector is split into four equal quarters:
      [0 : D/4]     → row-rotation first  half (via rotate-half within D/2 row block)
      [D/4 : D/2]   → row-rotation second half
      [D/2 : 3D/4]  → col-rotation first  half
      [3D/4 : D]    → col-rotation second half

    Args:
        x            : (B, N, D)
        positions_hw : (N, 2) int64 — (row_idx, col_idx) per token
        row_cos, row_sin : (max_h, D // 4)
        col_cos, col_sin : (max_w, D // 4)

    Returns:
        (B, N, D) rotated tensor
    """
    D   = x.shape[-1]
    q   = D // 4

    row_idx = positions_hw[:, 0]     # (N,)
    col_idx = positions_hw[:, 1]     # (N,)

    rcos = row_cos[row_idx]          # (N, q)
    rsin = row_sin[row_idx]
    ccos = col_cos[col_idx]
    csin = col_sin[col_idx]

    # Row rotation — first  D/2 dimensions
    x_r1     = x[..., :q]           # (B, N, q)
    x_r2     = x[..., q : 2 * q]
    row_out  = torch.cat([
        x_r1 * rcos - x_r2 * rsin,
        x_r2 * rcos + x_r1 * rsin,
    ], dim=-1)                       # (B, N, D/2)

    # Column rotation — last D/2 dimensions
    x_c1     = x[..., 2 * q : 3 * q]
    x_c2     = x[..., 3 * q:]
    col_out  = torch.cat([
        x_c1 * ccos - x_c2 * csin,
        x_c2 * ccos + x_c1 * csin,
    ], dim=-1)                       # (B, N, D/2)

    return torch.cat([row_out, col_out], dim=-1)


# =============================================================================
# Internal helpers
# =============================================================================

def _poly_features(
    x: torch.Tensor,
    coeff: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Polynomial expansion without aggregation.

    Equivalent to the PyTorch branch of polynomial_aggregation_ but returns
    per-token features (B, N, D) instead of the aggregated (B, G, D).

    Args:
        x     : (B, N, D) projected context (before activation)
        coeff : (D, K)    polynomial coefficients
        k     : degree

    Returns:
        (B, N, D) per-token polynomial features
    """
    h = pom_activation(x).unsqueeze(-1)    # (B, N, D, 1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    return (torch.cat(powers, dim=-1) * coeff).sum(-1)  # (B, N, D)


def _aggregate(h: torch.Tensor, mask) -> torch.Tensor:
    """Aggregate (B, N, D) polynomial features according to mask.

    Mirrors the mask-dispatch logic in polynomial_aggregation_ but operates
    on already-computed per-token features (no activation / poly expansion).

    Returns:
        (B, 1, D)  for mask=None or 2-D mask
        (B, N, D)  for mask="causal"
        (B, M, D)  for 3-D mask
    """
    if mask is None:
        return h.mean(dim=1, keepdim=True)

    if mask == "causal":
        B, N, _ = h.shape
        tril = torch.tril(torch.ones(N, N, device=h.device, dtype=h.dtype))
        return full_mask_mixer(h, tril.unsqueeze(0).expand(B, -1, -1))

    if isinstance(mask, torch.Tensor):
        if mask.dim() == 2:
            return mask_mixer(h, mask.to(h.device))
        if mask.dim() == 3:
            return full_mask_mixer(h, mask.to(h.device))

    raise ValueError(
        f'Unsupported mask: expected None, "causal", '
        f'or a 2/3-D tensor, got {mask!r}.'
    )


# =============================================================================
# PoMRoPE module
# =============================================================================

class PoMRoPE(PoM):
    """Polynomial Mixer with Rotary Position Embedding (RoPE).

    Extends PoM by injecting RoPE *after* the polynomial non-linearity and
    *before* the element-wise gating:

        H̃_j = R_j · poly(act(W_c X_j))      — rotated per-context-token
        S̃_i = R_i · gate(W_s X_i)            — rotated per-query-token
                                                (frequency-aligned for n_sel_heads > 1)
        y_i  = ag_proj(S̃_i ⊙ agg(H̃, mask))

    See module docstring for the mathematical justification.

    Args:
        dim, degree, expand, n_groups, n_sel_heads, bias
            Identical to PoM.

        max_seq_len : maximum 1-D sequence length.  Ignored when rope_2d=True.
        max_h, max_w: maximum grid height / width for 2-D RoPE.
        rope_base   : base for the frequency schedule (default 10 000).
        rope_2d     : if True, use 2-D RoPE (row + column positions).
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        expand: int,
        n_groups: int,
        n_sel_heads: int,
        bias: bool = False,
        max_seq_len: int = 4096,
        max_h: int = 64,
        max_w: int = 64,
        rope_base: float = 10000.0,
        rope_2d: bool = False,
    ):
        super().__init__(dim, degree, expand, n_groups, n_sel_heads, bias)
        self.rope_2d  = rope_2d
        rope_dim      = dim * expand          # dimension to which RoPE is applied

        if n_sel_heads > 1:
            hd = rope_dim // n_sel_heads
            if rope_2d:
                assert n_sel_heads % 4 == 0, (
                    f"n_sel_heads ({n_sel_heads}) must be divisible by 4 for 2-D RoPE"
                )
                assert hd >= 4, (
                    f"head_dim ({hd}) must be ≥ 4 for 2-D RoPE with n_sel_heads > 1"
                )
            else:
                assert n_sel_heads % 2 == 0, (
                    f"n_sel_heads ({n_sel_heads}) must be even for 1-D RoPE"
                )
                assert hd >= 2, (
                    f"head_dim ({hd}) must be ≥ 2 for 1-D RoPE with n_sel_heads > 1"
                )

        if rope_2d:
            assert rope_dim % 4 == 0, (
                f"dim * expand ({rope_dim}) must be divisible by 4 for 2-D RoPE"
            )
            row_cos, row_sin, col_cos, col_sin = precompute_freqs_2d(
                rope_dim, max_h, max_w, rope_base
            )
            self.register_buffer('row_cos', row_cos)   # (max_h, rope_dim // 4)
            self.register_buffer('row_sin', row_sin)
            self.register_buffer('col_cos', col_cos)   # (max_w, rope_dim // 4)
            self.register_buffer('col_sin', col_sin)
        else:
            assert rope_dim % 2 == 0, (
                f"dim * expand ({rope_dim}) must be even for 1-D RoPE"
            )
            freqs_cos, freqs_sin = precompute_freqs_1d(
                rope_dim, max_seq_len, rope_base
            )
            self.register_buffer('freqs_cos', freqs_cos)  # (max_seq_len, rope_dim // 2)
            self.register_buffer('freqs_sin', freqs_sin)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _rope(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor],
        positions_hw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply 1-D or 2-D RoPE based on module configuration.

        For 1-D, if positions is None the token indices 0..N-1 are used.
        For 2-D, positions_hw must be supplied.

        When x has fewer channels than the full polynomial dimension (i.e. x
        is the n_sel_heads-dimensional selection signal), the frequency tables
        are subsampled so that the k-th dimension pair of S shares the leading
        frequency of head k in H.
        """
        N    = x.shape[1]
        D    = x.shape[-1]
        full = self._po_dim   # frequency tables were built for this dimension

        if self.rope_2d:
            if positions_hw is None:
                raise ValueError(
                    "positions_hw (N, 2) must be provided when rope_2d=True"
                )
            if D == full:
                return apply_rope_2d(
                    x, positions_hw,
                    self.row_cos, self.row_sin,
                    self.col_cos, self.col_sin,
                )
            # s path: D = n_sel_heads.  stride = head_dim selects the leading
            # frequency of each head from the (rope_dim // 4)-column tables.
            stride = self.head_dim
            q      = D // 4
            rcos   = self.row_cos[:, ::stride][:, :q]
            rsin   = self.row_sin[:, ::stride][:, :q]
            ccos   = self.col_cos[:, ::stride][:, :q]
            csin   = self.col_sin[:, ::stride][:, :q]
            return apply_rope_2d(x, positions_hw, rcos, rsin, ccos, csin)
        else:
            if positions is None:
                positions = torch.arange(N, device=x.device)
            if D == full:
                return apply_rope_1d(x, positions, self.freqs_cos, self.freqs_sin)
            # s path: D = n_sel_heads.  Pair k uses freq index k * (head_dim//2),
            # the leading frequency band of head k in H.
            stride = self.head_dim // 2
            half   = D // 2
            cos    = self.freqs_cos[:, ::stride][:, :half]
            sin    = self.freqs_sin[:, ::stride][:, :half]
            return apply_rope_1d(x, positions, cos, sin)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        xq: torch.Tensor,
        xc: Optional[torch.Tensor] = None,
        mask=None,
        positions: Optional[torch.Tensor] = None,
        ctx_positions: Optional[torch.Tensor] = None,
        positions_hw: Optional[torch.Tensor] = None,
        ctx_positions_hw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            xq               : (B, T, D) query tokens.
            xc               : (B, N, D) context tokens; defaults to xq
                               (self-mixing).
            mask             : None | "causal" | (B, N) | (B, M, N).
            positions        : (T,) int64 query token positions for 1-D RoPE.
                               Defaults to 0..T-1.
            ctx_positions    : (N,) int64 context token positions for 1-D
                               RoPE.  Defaults to `positions` when xq is xc,
                               otherwise 0..N-1.
            positions_hw     : (T, 2) int64 query (row, col) for 2-D RoPE.
            ctx_positions_hw : (N, 2) int64 context (row, col) for 2-D RoPE.
                               Defaults to `positions_hw` when xq is xc.

        Returns:
            (B, max(T, G), D) where G is determined by the mask type.
        """
        self_mixing = xc is None or xc is xq
        if xc is None:
            xc = xq

        # ---- projections ------------------------------------------------
        h_proj, s = self._get_h_s(xq, xc)
        # h_proj : (B, N, D_po)  — linear features for context (pre-activation)
        # s      : (B, T, D_se)  — hardsigmoid gating for query

        # ---- resolve context positions ----------------------------------
        eff_ctx_pos    = ctx_positions    if ctx_positions    is not None else (positions    if self_mixing else None)
        eff_ctx_pos_hw = ctx_positions_hw if ctx_positions_hw is not None else (positions_hw if self_mixing else None)

        # ---- fused poly + RoPE + aggregate (Triton fast paths) ----------
        # Conditions: CUDA tensor, 1-D RoPE, unmasked or causal mask.
        # 2-D RoPE and other mask types fall through to the PyTorch path.
        _use_triton = (
            TRITON_ROPE_AVAILABLE
            and h_proj.is_cuda
            and not self.rope_2d
            and positions_hw is None
        )
        if _use_triton and mask is None:
            ctx_pos = eff_ctx_pos
            if ctx_pos is None:
                ctx_pos = torch.arange(h_proj.shape[1], device=h_proj.device,
                                       dtype=torch.int64)
            h_agg = poly_agg_rope_mean_triton(
                h_proj, self.po_coeff, self.order,
                self.freqs_cos, self.freqs_sin, ctx_pos,
            )
        elif _use_triton and mask == "causal":
            ctx_pos = eff_ctx_pos
            if ctx_pos is None:
                ctx_pos = torch.arange(h_proj.shape[1], device=h_proj.device,
                                       dtype=torch.int64)
            h_agg = poly_agg_rope_causal_triton(
                h_proj, self.po_coeff, self.order,
                self.freqs_cos, self.freqs_sin, ctx_pos,
            )
        else:
            # PyTorch fallback: poly → RoPE → aggregate
            h = _poly_features(h_proj, self.po_coeff, self.order)
            h = self._rope(h, eff_ctx_pos, eff_ctx_pos_hw)
            h_agg = _aggregate(h, mask)

        # ---- apply RoPE to query selection signal -----------------------
        # For n_sel_heads ≤ 1: s has the full D_po channels → standard RoPE.
        # For n_sel_heads > 1: s has n_sel_heads channels; _rope subsamples
        # the frequency tables so each scalar gate aligns with its head's
        # leading frequency in H.
        s = self._rope(s, positions, positions_hw)

        # ---- select and project -----------------------------------------
        sh = polynomial_selection_(s, h_agg, self.n_sel_heads)
        return self.ag_proj(sh)
