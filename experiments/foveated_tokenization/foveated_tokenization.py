"""
Foveated Tokenization — STT (CVPR 2025) 스타일 구현.

Bio-inspired foveated vision: 중심와 근처는 고해상도, 주변부는 저해상도.

3-level concentric ring 구조:
  Level 1 (center) : stride=1, 4×4 grid → 16×16px 커버, 전체 해상도
  Level 2 (mid)    : stride=2, 4×4 grid → 32×32px → 16×16 downsample
  Level 3 (periph) : stride=4, 4×4 grid → 64×64px → 16×16 downsample

각 level에서 내부 level과 겹치는 inner 2×2 patches 제거.
결과: Level1 16개 + Level2 12개 + Level3 12개 = 40 patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

PATCH_OUT = 16   # 모든 output patch 크기
GRID      = 4    # 각 level의 grid 크기

# (stride, output_patch_size) 형태로 정의
LEVELS = [
    {"stride": 1, "label": "L1-center"},
    {"stride": 2, "label": "L2-mid"},
    {"stride": 4, "label": "L3-periph"},
]

# Level별 색상 (BGR for OpenCV, RGB for matplotlib)
LEVEL_COLORS_RGB = [
    (255,  80,  80),   # L1: red
    ( 80, 200,  80),   # L2: green
    ( 80, 120, 255),   # L3: blue
]


@dataclass
class FovPatch:
    """하나의 foveated patch 정보."""
    patch: np.ndarray   # (PATCH_OUT, PATCH_OUT, 3) uint8
    orig_x0: int        # original 이미지 기준 left
    orig_y0: int        # original 이미지 기준 top
    orig_size: int      # original 이미지 기준 커버 크기 (stride * PATCH_OUT)
    stride: int         # 1, 2, 4
    level: int          # 0, 1, 2
    row: int            # grid row (0..3)
    col: int            # grid col (0..3)


def _is_inner(row: int, col: int) -> bool:
    """4×4 grid에서 inner 2×2 여부 (row∈{1,2}, col∈{1,2})."""
    return 1 <= row <= 2 and 1 <= col <= 2


def _extract_patch(
    image: np.ndarray,
    x0: int, y0: int,
    size: int,
    out_size: int,
) -> np.ndarray:
    """
    image에서 (x0,y0) 기준 size×size 영역을 잘라 out_size×out_size로 리사이즈.
    image 밖은 0-padding.
    """
    H, W = image.shape[:2]

    # 실제 이미지 안에서 clamp된 좌표
    x1, y1 = x0 + size, y0 + size
    pad_l = max(0, -x0);  pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)

    if cx0 >= cx1 or cy0 >= cy1:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    crop = image[cy0:cy1, cx0:cx1].copy()
    if pad_l or pad_t or pad_r or pad_b:
        crop = np.pad(
            crop,
            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    if size != out_size:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.uint8)


def foveated_tokenize(
    image: np.ndarray,
    cx: int,
    cy: int,
) -> List[FovPatch]:
    """
    이미지에 foveated tokenization 적용.

    Args:
        image : (H, W, 3) uint8 RGB 이미지.
        cx, cy: foveation center (pixel 좌표, 0-indexed).

    Returns:
        FovPatch 리스트 (총 40개; 이미지 밖 patch는 제외될 수 있음).
    """
    patches: List[FovPatch] = []

    for lvl_idx, lvl in enumerate(LEVELS):
        stride   = lvl["stride"]
        coverage = PATCH_OUT * stride   # 원본 이미지에서 각 patch가 커버하는 픽셀 수

        # 4×4 grid를 cx,cy 중심으로 배치
        total  = GRID * coverage           # 전체 커버 범위
        x0_grid = cx - total // 2
        y0_grid = cy - total // 2

        for row in range(GRID):
            for col in range(GRID):
                # inner 2×2는 더 안쪽 level이 이미 커버 → 제거
                if lvl_idx > 0 and _is_inner(row, col):
                    continue

                ox0 = x0_grid + col * coverage
                oy0 = y0_grid + row * coverage

                patch_img = _extract_patch(image, ox0, oy0, coverage, PATCH_OUT)

                patches.append(FovPatch(
                    patch=patch_img,
                    orig_x0=ox0,
                    orig_y0=oy0,
                    orig_size=coverage,
                    stride=stride,
                    level=lvl_idx,
                    row=row,
                    col=col,
                ))

    return patches


# ─────────────────────────────────────────
# Utility: patch list → numpy token array
# ─────────────────────────────────────────

def patches_to_array(patches: List[FovPatch]) -> np.ndarray:
    """(N, PATCH_OUT, PATCH_OUT, 3) uint8 배열로 변환."""
    return np.stack([p.patch for p in patches], axis=0)


def patches_to_coords(patches: List[FovPatch]) -> np.ndarray:
    """각 patch의 (orig_x_center, orig_y_center, stride) → (N, 3) 배열."""
    coords = []
    for p in patches:
        xc = p.orig_x0 + p.orig_size // 2
        yc = p.orig_y0 + p.orig_size // 2
        coords.append([xc, yc, p.stride])
    return np.array(coords, dtype=np.float32)


# ─────────────────────────────────────────
# Utility: reconstruct 원본 해상도 이미지
# ─────────────────────────────────────────

def reconstruct_foveated(
    patches: List[FovPatch],
    canvas_shape: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    Patches를 원래 위치에 upsample하여 canvas에 그림.
    바깥 level일수록 먼저 그려서 안쪽이 위에 덮임 (고해상도가 최상위).

    Args:
        canvas_shape: (H, W) 출력 canvas 크기.
        bg_color    : 배경색 (RGB).
    """
    H, W = canvas_shape
    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    # 바깥 → 안쪽 순서로 그리기 (level 내림차순)
    for p in sorted(patches, key=lambda x: -x.level):
        x0, y0 = p.orig_x0, p.orig_y0
        x1, y1 = x0 + p.orig_size, y0 + p.orig_size

        # canvas 밖이면 skip
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue

        # patch를 orig_size×orig_size로 upsample
        up = cv2.resize(p.patch, (p.orig_size, p.orig_size), interpolation=cv2.INTER_NEAREST)

        # canvas 경계에 맞춰 clamp
        src_x0 = max(0, -x0);    src_y0 = max(0, -y0)
        src_x1 = p.orig_size - max(0, x1 - W)
        src_y1 = p.orig_size - max(0, y1 - H)

        dst_x0 = max(0, x0);    dst_y0 = max(0, y0)
        dst_x1 = min(W, x1);    dst_y1 = min(H, y1)

        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = up[src_y0:src_y1, src_x0:src_x1]

    return canvas
