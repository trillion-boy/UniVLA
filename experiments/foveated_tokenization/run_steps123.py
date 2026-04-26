"""
Foveated Tokenization 초기 실험 — Step 1~3 통합 스크립트.

Step 1: SimplerEnv WidowX task에서 이미지 + GT object position 추출
Step 2: Foveated tokenization 적용 (foveated_tokenization.py)
Step 3: 3-panel 시각화 저장
  (a) 원본 이미지 + fovea center + level 경계 사각형
  (b) 추출된 foveated patches grid (40 patches)
  (c) Reconstructed view (patches를 원본 크기로 복원)

사용법 (실제 실행 환경 /content/ 기준):
  cd /content/UniVLA
  python experiments/foveated_tokenization/run_steps123.py \
      --task widowx_put_eggplant_in_basket \
      --n-samples 3 \
      --output-dir /content/foveated_exp
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────
# 경로 설정 (실행 환경 /content/ 기준)
# ──────────────────────────────────────────
SIMPLER_ENV_ROOT   = "/content/SimplerEnv"
MANISKILL2_ROOT    = os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim")
UNIVLA_ROOT        = "/content/UniVLA"
EXPERIMENT_DIR     = os.path.join(UNIVLA_ROOT, "experiments", "foveated_tokenization")

for p in [SIMPLER_ENV_ROOT, MANISKILL2_ROOT, UNIVLA_ROOT, EXPERIMENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from foveated_tokenization import (   # noqa: E402
    LEVEL_COLORS_RGB,
    LEVELS,
    FovPatch,
    foveated_tokenize,
    patches_to_array,
    reconstruct_foveated,
    reconstruct_foveated_with_context,
    PATCH_OUT,
    GRID,
)


# ════════════════════════════════════════════
# Step 1: SimplerEnv 데이터 추출
# ════════════════════════════════════════════

WIDOWX_CAMERA = "3rd_view_camera"

# task → source object 접근 방식
_TASK_OBJ_ATTR = {
    "widowx_put_eggplant_in_basket": "episode_source_obj",
    "widowx_carrot_on_plate":        "episode_source_obj",
    "widowx_spoon_on_towel":         "episode_source_obj",
    "widowx_stack_cube":             "episode_source_obj",
}


def _get_manip_obj(env, task: str):
    """환경에서 manipulation target object 반환."""
    attr = _TASK_OBJ_ATTR.get(task, "obj")
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, "obj"):
        return env.obj
    raise AttributeError(f"Cannot find manipulation object in env for task={task}")


def project_3d_to_2d(
    point_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    """
    3D world point → 2D pixel coordinate.

    Args:
        point_3d : (3,) world coords
        intrinsic: (3,3) OpenCV intrinsic
        extrinsic: (4,4) world-to-camera extrinsic (OpenCV convention)

    Returns:
        (2,) pixel [u, v]
    """
    ph = np.array([*point_3d, 1.0])   # homogeneous (4,)
    p_cam = extrinsic @ ph             # (4,) camera coords
    p_cam = p_cam[:3]
    p_img = intrinsic @ p_cam          # (3,)
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    return np.array([u, v])


def extract_samples(task: str, n: int) -> list[dict]:
    """
    SimplerEnv에서 n번 reset하며 이미지 + GT object 2D 좌표 수집.
    각 sample: {"image": (H,W,3), "pixel_xy": (2,), "obj_3d": (3,)}
    """
    import simpler_env   # noqa: PLC0415
    import gymnasium as gym   # noqa: PLC0415, F401
    import mani_skill2_real2sim.envs   # noqa: PLC0415, F401

    print(f"[Step1] Making env: {task}")
    env = simpler_env.make(task)

    samples = []
    for i in range(n):
        obs, _ = env.reset()

        # 이미지 가져오기
        cam_obs = obs["image"].get(WIDOWX_CAMERA)
        if cam_obs is None:
            available = list(obs["image"].keys())
            print(f"  [WARN] '{WIDOWX_CAMERA}' not found; using first: {available[0]}")
            cam_obs = obs["image"][available[0]]
            cam_key = available[0]
        else:
            cam_key = WIDOWX_CAMERA

        # rgb: (H, W, 3) uint8 or float
        img = cam_obs.get("rgb", None)
        if img is None:
            # Color is RGBA float [0,1]
            img = (cam_obs["Color"][..., :3] * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # gymnasium wrapper가 private attr를 막으므로 unwrapped로 접근
        raw_env = env.unwrapped

        # 카메라 파라미터
        cam_params = raw_env._cameras[cam_key].get_params()
        intrinsic  = cam_params["intrinsic_cv"]   # (3,3)
        extrinsic  = cam_params["extrinsic_cv"]   # (4,4)

        # GT object 위치
        obj      = _get_manip_obj(raw_env, task)
        obj_3d   = np.array(obj.pose.p)           # (3,)
        pixel_xy = project_3d_to_2d(obj_3d, intrinsic, extrinsic).astype(int)

        H, W = img.shape[:2]
        pixel_xy = np.clip(pixel_xy, [0, 0], [W - 1, H - 1])

        samples.append({
            "image":    img,
            "pixel_xy": pixel_xy,
            "obj_3d":   obj_3d,
        })
        print(f"  sample {i}: obj_3d={obj_3d}, pixel_xy={pixel_xy}, img_shape={img.shape}")

    env.close()
    return samples


# ════════════════════════════════════════════
# Step 3: 시각화
# ════════════════════════════════════════════

def _level_rect_in_image(cx: int, cy: int, lvl_idx: int):
    """level_idx에 해당하는 bounding box (x0, y0, w, h) 반환 (원본 이미지 좌표)."""
    stride   = LEVELS[lvl_idx]["stride"]
    coverage = PATCH_OUT * stride
    total    = GRID * coverage
    x0 = cx - total // 2
    y0 = cy - total // 2
    return x0, y0, total, total


def visualize_sample(
    sample: dict,
    sample_idx: int,
    output_dir: str,
) -> str:
    """
    3-panel 시각화 생성 & 저장.
    Returns saved path.
    """
    img      = sample["image"]         # (H, W, 3) uint8
    cx, cy   = sample["pixel_xy"]
    H, W     = img.shape[:2]

    # ── Foveated tokenization ──
    patches  = foveated_tokenize(img, int(cx), int(cy))
    # 배경: 원본 blurry → 위에 foveated patches 덮기 (주변 환경 맥락 유지)
    recon    = reconstruct_foveated_with_context(img, patches)
    tokens   = patches_to_array(patches)   # (N, 16, 16, 3)
    N        = len(patches)

    # ── Figure 구성 ──
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(
        f"Foveated Tokenization — sample {sample_idx}  |  "
        f"fovea=({cx},{cy})  |  {N} patches",
        fontsize=13, y=1.01,
    )

    # ──────────────────────────
    # Panel (a): 원본 이미지
    # ──────────────────────────
    ax_orig = fig.add_subplot(1, 3, 1)
    ax_orig.imshow(img)
    ax_orig.set_title("(a) Original + fovea center", fontsize=11)

    # fovea center 마커
    ax_orig.plot(cx, cy, "y*", markersize=14, zorder=10)

    # 각 level bounding box
    for lvl_idx, lvl in enumerate(LEVELS):
        bx, by, bw, bh = _level_rect_in_image(cx, cy, lvl_idx)
        color = [c / 255 for c in LEVEL_COLORS_RGB[lvl_idx]]
        rect  = mpatches.Rectangle(
            (bx, by), bw, bh,
            linewidth=2, edgecolor=color, facecolor="none",
            linestyle=["solid", "dashed", "dotted"][lvl_idx],
            zorder=5,
        )
        ax_orig.add_patch(rect)

    # legend
    legend_handles = [
        mpatches.Patch(color=[c / 255 for c in LEVEL_COLORS_RGB[i]],
                       label=f"L{i+1} stride={LEVELS[i]['stride']} ({['16px','32→16px','64→16px'][i]})")
        for i in range(len(LEVELS))
    ]
    ax_orig.legend(handles=legend_handles, loc="upper left", fontsize=8,
                   framealpha=0.8)
    ax_orig.axis("off")

    # ──────────────────────────
    # Panel (b): Patches grid
    # ──────────────────────────
    ax_patches = fig.add_subplot(1, 3, 2)
    ax_patches.set_title(f"(b) Foveated patches ({N} total)", fontsize=11)

    # N patches를 정사각에 가까운 그리드로 배치
    n_cols = 8
    n_rows = int(np.ceil(N / n_cols))
    canvas_b = np.zeros((n_rows * PATCH_OUT, n_cols * PATCH_OUT, 3), dtype=np.uint8)

    for idx, p in enumerate(patches):
        r, c = divmod(idx, n_cols)
        y0b  = r * PATCH_OUT
        x0b  = c * PATCH_OUT
        canvas_b[y0b:y0b + PATCH_OUT, x0b:x0b + PATCH_OUT] = p.patch

        # border color per level
        bc = LEVEL_COLORS_RGB[p.level]
        canvas_b[y0b, x0b:x0b + PATCH_OUT]               = bc  # top
        canvas_b[y0b + PATCH_OUT - 1, x0b:x0b + PATCH_OUT] = bc  # bottom
        canvas_b[y0b:y0b + PATCH_OUT, x0b]               = bc  # left
        canvas_b[y0b:y0b + PATCH_OUT, x0b + PATCH_OUT - 1] = bc  # right

    ax_patches.imshow(canvas_b)
    ax_patches.set_xlabel(
        "Red=L1(stride1)  Green=L2(stride2)  Blue=L3(stride4)",
        fontsize=8,
    )
    ax_patches.axis("off")

    # ──────────────────────────
    # Panel (c): Reconstruction
    # ──────────────────────────
    ax_recon = fig.add_subplot(1, 3, 3)
    ax_recon.imshow(recon)
    ax_recon.set_title("(c) Reconstructed view", fontsize=11)
    ax_recon.plot(cx, cy, "y*", markersize=14, zorder=10)

    # level 경계 다시 표시
    for lvl_idx in range(len(LEVELS)):
        bx, by, bw, bh = _level_rect_in_image(cx, cy, lvl_idx)
        color = [c / 255 for c in LEVEL_COLORS_RGB[lvl_idx]]
        rect  = mpatches.Rectangle(
            (bx, by), bw, bh,
            linewidth=1.5, edgecolor=color, facecolor="none",
            linestyle=["solid", "dashed", "dotted"][lvl_idx],
            zorder=5,
        )
        ax_recon.add_patch(rect)
    ax_recon.axis("off")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"foveated_sample_{sample_idx:02d}.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Step3] Saved visualization → {save_path}")
    return save_path


def visualize_patch_layout(
    sample: dict,
    sample_idx: int,
    output_dir: str,
) -> str:
    """
    추가 시각화: 각 patch의 원본 이미지 내 위치를 색으로 표시.
    """
    img    = sample["image"].copy()
    cx, cy = sample["pixel_xy"]
    patches = foveated_tokenize(img, int(cx), int(cy))

    import cv2 as cv2_local
    vis = img.copy()

    for p in patches:
        color = LEVEL_COLORS_RGB[p.level]
        x0, y0 = p.orig_x0, p.orig_y0
        x1, y1 = x0 + p.orig_size, y0 + p.orig_size
        cv2_local.rectangle(vis, (x0, y0), (x1, y1), color, 1)

    # fovea center
    cv2_local.drawMarker(vis, (cx, cy), (255, 230, 0),
                         cv2_local.MARKER_STAR, 20, 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(vis)
    ax.set_title(f"Patch layout on original image — sample {sample_idx}", fontsize=11)
    legend_handles = [
        mpatches.Patch(color=[c / 255 for c in LEVEL_COLORS_RGB[i]],
                       label=f"L{i+1} (stride {LEVELS[i]['stride']})")
        for i in range(len(LEVELS))
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"foveated_layout_{sample_idx:02d}.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Step3] Saved layout vis → {save_path}")
    return save_path


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Foveated tokenization experiment steps 1-3")
    p.add_argument("--task", default="widowx_put_eggplant_in_basket",
                   choices=[
                       "widowx_put_eggplant_in_basket",
                       "widowx_carrot_on_plate",
                       "widowx_spoon_on_towel",
                       "widowx_stack_cube",
                   ])
    p.add_argument("--n-samples", type=int, default=3,
                   help="환경 reset 횟수 (각 reset마다 시각화 생성)")
    p.add_argument("--output-dir", default="/content/foveated_exp")
    p.add_argument(
        "--dummy-image",
        action="store_true",
        help="SimplerEnv 없이 더미 이미지로 foveated tokenization만 테스트",
    )
    return p.parse_args()


def _dummy_sample(H: int = 256, W: int = 256) -> dict:
    """SimplerEnv 없이 테스트용 더미 샘플 생성."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    # 가운데에 밝은 원 추가 (object 대용)
    import cv2 as _cv2
    _cv2.circle(img, (W // 2, H // 2), 20, (255, 180, 60), -1)
    return {"image": img, "pixel_xy": np.array([W // 2, H // 2]), "obj_3d": None}


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1 ──
    if args.dummy_image:
        print("[Step1] Using dummy image (SimplerEnv skipped)")
        samples = [_dummy_sample() for _ in range(args.n_samples)]
    else:
        samples = extract_samples(args.task, args.n_samples)

    # ── Step 2 + 3 ──
    print(f"\n[Step2/3] Running foveated tokenization & visualization...")
    saved = []
    for i, sample in enumerate(samples):
        main_path   = visualize_sample(sample, i, args.output_dir)
        layout_path = visualize_patch_layout(sample, i, args.output_dir)
        saved.extend([main_path, layout_path])

    print(f"\n Done. {len(saved)} files saved to {args.output_dir}/")
    for f in saved:
        print(f"  {f}")
