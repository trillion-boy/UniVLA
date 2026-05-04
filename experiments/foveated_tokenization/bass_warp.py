"""
bass_warp.py
Bio-inspired Adaptive Stereographic Sampling (BASS) — core warping module.

Theory
------
The Möbius transformation T: ℂ → ℂ,

    T(w) = s · (w − p) / (1 − p̄·w)            ... (1)

where p ∈ ℂ is the *pole* (fovea center) and s ∈ ℝ is the *strength*,
maps the pole p to the origin.  The inverse,

    T⁻¹(z) = (z/s + p) / (1 + p̄·z/s)          ... (2)

is what we apply to the *output* coordinate grid: for each output pixel z,
the source pixel in the input image is w = T⁻¹(z).  Because T⁻¹ maps a
small neighborhood of 0 to a small neighborhood of p in the input, the pole
region is *magnified* in the output.

Jacobian of T⁻¹ at z = 0:
    |dw/dz|₀ = (1 − |p|²) / s

Large s  →  small Jacobian  →  output pixels near 0 sample a tiny input
region  →  high effective resolution at the pole.

Pipeline (foveated VLA)
-----------------------
1. DINO detects fovea center(s)           → pole_px
2. Phase-aware strength                   → s
3. MobiusWarpModule.warp(image, pole, s)  → warped image (same HxW)
4. VQ-VAE encodes warped image            → 1024 tokens (pole region high-res)
5. (Optional) cpe_grid()                  → original pixel coords per token
                                            for action coordinate alignment

Advantages over crop-and-paste (TrueFoveated)
---------------------------------------------
- Single image through VQ-VAE: NO mixed-token OOD artefacts
- Full FOV preserved: periphery compressed, not cropped out
- Mathematically well-defined everywhere (no boundary discontinuities)
- Dual-focus: DualFocusOptimizer picks midpoint + reduced strength so both
  source and destination objects stay in the magnified zone
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Möbius primitives
# ══════════════════════════════════════════════════════════════════════════════

def _build_grid(H: int, W: int, device) -> torch.Tensor:
    """Normalised output coordinate grid (H, W) complex ∈ [−1, 1]²."""
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.complex(gx.float(), gy.float())


def _px_to_pole(cx: int, cy: int, W: int, H: int, device) -> torch.Tensor:
    """
    Pixel (cx, cy) → complex pole in [−1, 1]², clamped to |p| ≤ 0.70.

    The singularity of T⁻¹ lies at |z| = s/|p|. With |p| ≤ 0.70 and s ≥ 1
    the singularity is at |z| ≥ 1/0.70 ≈ 1.43, safely outside the image
    corners (|z|_max = √2 ≈ 1.41 — just at the limit; 0.70 gives a margin).
    """
    x = float(cx) / W * 2.0 - 1.0
    y = float(cy) / H * 2.0 - 1.0
    pole = torch.complex(
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )
    mag = pole.abs().item()
    if mag > 0.70:
        pole = pole * (0.70 / mag)
    return pole


def _mobius_inverse(z: torch.Tensor, pole: torch.Tensor, s: float) -> torch.Tensor:
    """
    T⁻¹(z) = (z/s + p) / (1 + p̄·z/s)

    Both z and pole are complex tensors.  s is a real Python float.
    """
    z_s = z / s
    denom = 1.0 + pole.conj() * z_s
    # Guard against near-zero denominators (shouldn't occur with |pole| ≤ 0.70
    # but kept for robustness under edge cases).
    safe_denom = torch.where(
        denom.abs() < 1e-5,
        torch.tensor(1e-5 + 0j, dtype=denom.dtype, device=denom.device),
        denom,
    )
    return (z_s + pole) / safe_denom


# ══════════════════════════════════════════════════════════════════════════════
# MobiusWarpModule
# ══════════════════════════════════════════════════════════════════════════════

class MobiusWarpModule(nn.Module):
    """
    Foveated image warping via Möbius transformation.

    The warped image has the pole region magnified (more output pixels per
    input pixel area) while the periphery is compressed — but never removed.
    Feeding the warped image to VQ-VAE produces higher-resolution tokens at
    the pole with no OOD artefacts (single coherent image, no token mixing).

    Parameters
    ----------
    resolution_floor : float
        Minimum downscale factor when using sample_budget().  Prevents the
        periphery from shrinking to nothing.  Default 0.20 (= 20 %).
    """

    def __init__(self, resolution_floor: float = 0.20):
        super().__init__()
        self.resolution_floor = resolution_floor

    # ── core warp ─────────────────────────────────────────────────────────────

    def warp(
        self,
        image: torch.Tensor,
        focus_px: Tuple[int, int],
        strength: float,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply foveated Möbius warp: pole region is magnified in the output.

        Parameters
        ----------
        image    : (B, C, H, W) float tensor, values in [0, 1]
        focus_px : (cx, cy) fovea centre in pixel coordinates of the INPUT
        strength : magnification s > 1  (s = 1 → identity warp)
        out_hw   : optional (H_out, W_out); defaults to input size

        Returns
        -------
        warped : (B, C, H_out, W_out) float tensor
        """
        B, C, H, W = image.shape
        H_out, W_out = out_hw if out_hw else (H, W)
        dev = image.device

        pole = _px_to_pole(*focus_px, W, H, dev)       # complex scalar
        z    = _build_grid(H_out, W_out, dev)           # (H_out, W_out) complex
        w    = _mobius_inverse(z, pole, strength)       # source coords, complex

        # grid_sample expects (B, H, W, 2) with (x, y) order, values in [−1,1]
        grid = torch.stack(
            [w.real.clamp(-1.0, 1.0), w.imag.clamp(-1.0, 1.0)], dim=-1
        ).unsqueeze(0).expand(B, -1, -1, -1)           # (B, H_out, W_out, 2)

        return F.grid_sample(
            image, grid,
            mode="bilinear",
            padding_mode="border",   # repeat border for out-of-range samples
            align_corners=True,
        )

    # ── continuous position encoding ──────────────────────────────────────────

    def cpe_grid(
        self,
        focus_px: Tuple[int, int],
        strength: float,
        H_out: int, W_out: int,
        H_in:  int, W_in:  int,
        device,
    ) -> torch.Tensor:
        """
        Continuous Position Encoding (CPE): for every position in the *warped*
        output grid, return the original normalised pixel coordinate (u, v).

        The CPE can be used to re-align the spatial semantics of warped tokens
        with the robot's action coordinate system (which is always in the
        original, un-warped frame).  Concretely: concatenate or add a
        2-channel CPE map to the token feature map before action decoding.

        Returns
        -------
        cpe : (H_out, W_out, 2) — (u_orig, v_orig) normalised to [0, 1]
              u_orig = 0 → left edge,  u_orig = 1 → right edge (of original)
        """
        pole = _px_to_pole(*focus_px, W_in, H_in, device)
        z    = _build_grid(H_out, W_out, device)
        w    = _mobius_inverse(z, pole, strength)
        u = (w.real.clamp(-1.0, 1.0) + 1.0) * 0.5     # [0, 1]
        v = (w.imag.clamp(-1.0, 1.0) + 1.0) * 0.5
        return torch.stack([u, v], dim=-1)              # (H_out, W_out, 2)

    # ── pixel-budget uniform sampling ─────────────────────────────────────────

    def sample_budget(
        self,
        warped: torch.Tensor,
        budget: float,
    ) -> torch.Tensor:
        """
        Downsample the warped image to a target fraction of pixels.

        Because the pole region is already magnified in `warped`, a *uniform*
        spatial subsample still allocates more samples to the pole than to
        the periphery.  This achieves efficient token allocation without any
        non-uniform grid operations.

        The resolution_floor ensures the periphery retains at least a minimum
        number of pixels so peripheral information is not completely lost.

        Parameters
        ----------
        warped : (B, C, H, W)
        budget : fraction of pixels to keep, e.g. 0.10 = 10 %

        Returns
        -------
        sampled : (B, C, H_s, W_s)
        """
        B, C, H, W = warped.shape
        scale = max(math.sqrt(max(budget, 0.0)), self.resolution_floor)
        H_s = max(int(H * scale), 32)
        W_s = max(int(W * scale), 32)
        return F.interpolate(
            warped, (H_s, W_s), mode="bilinear", align_corners=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# Phase-aware magnification
# ══════════════════════════════════════════════════════════════════════════════

class PhaseAwareMagnification:
    """
    Adjusts Möbius warp strength based on the robot's manipulation phase.

    Grasping phase (gripper open)
        → high strength: dense tokens on the target object for precise grasping.

    Moving phase (gripper closed, carrying object)
        → low strength: wider context so the destination is clearly visible.

    Usage
    -----
    phase = PhaseAwareMagnification()
    phase.update(gripper_norm)   # call after every action step
    s = phase.strength           # use for this frame's warp
    """

    def __init__(
        self,
        grasp_strength: float = 4.0,
        move_strength:  float = 2.0,
        close_thresh:   float = 0.5,   # gripper_norm ≥ this → "closed"
    ):
        self.grasp_strength = grasp_strength
        self.move_strength  = move_strength
        self.close_thresh   = close_thresh
        self._gripper: float = 0.0     # 0 = open, 1 = fully closed

    def update(self, gripper_norm: float) -> None:
        """
        gripper_norm: 0.0 = fully open, 1.0 = fully closed.
        For WidowX: map env gripper action [-1, +1] → norm via (1 - g) / 2.
        """
        self._gripper = float(np.clip(gripper_norm, 0.0, 1.0))

    @property
    def phase(self) -> str:
        return "grasping" if self._gripper < self.close_thresh else "moving"

    @property
    def strength(self) -> float:
        return self.grasp_strength if self.phase == "grasping" else self.move_strength

    def __repr__(self) -> str:
        return (f"Phase({self.phase}, gripper={self._gripper:.2f}, "
                f"s={self.strength:.1f})")


# ══════════════════════════════════════════════════════════════════════════════
# Dual-focus optimiser
# ══════════════════════════════════════════════════════════════════════════════

class DualFocusOptimizer:
    """
    For two-object tasks (e.g. "stack green block on yellow block"): finds the
    Möbius pole and strength such that both source and destination objects lie
    within the magnified fovea region.

    Algorithm
    ---------
    Pole  = midpoint(src, dst)
    The fovea radius in normalised coords ≈ 1/s.
    To cover distance d_norm from the midpoint to each object:
        1/s ≥ d_norm  →  s ≤ 1/d_norm
    With a safety margin α:
        s ≤ α / d_norm
    Clamp to [min_strength, max_strength].
    """

    def __init__(
        self,
        max_strength:    float = 4.0,
        min_strength:    float = 1.5,
        coverage_margin: float = 0.9,  # α: < 1 ensures both objects in fovea
    ):
        self.max_strength    = max_strength
        self.min_strength    = min_strength
        self.coverage_margin = coverage_margin

    def compute(
        self,
        src_px: Tuple[int, int],
        dst_px: Tuple[int, int],
        W: int, H: int,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Returns
        -------
        pole_px  : (cx, cy) optimal focus point (midpoint of src and dst)
        strength : optimal magnification strength
        """
        cx = int((src_px[0] + dst_px[0]) / 2)
        cy = int((src_px[1] + dst_px[1]) / 2)

        # Half-distance from midpoint to either object, normalised by image diagonal
        d_norm = math.sqrt(
            ((src_px[0] - dst_px[0]) / W) ** 2
            + ((src_px[1] - dst_px[1]) / H) ** 2
        ) / 2.0

        if d_norm > 1e-3:
            s = self.coverage_margin / d_norm
        else:
            s = self.max_strength

        s = float(np.clip(s, self.min_strength, self.max_strength))
        print(f"[DualFocus] src={src_px} dst={dst_px} mid=({cx},{cy}) "
              f"d_norm={d_norm:.3f} → s={s:.2f}")
        return (cx, cy), s
