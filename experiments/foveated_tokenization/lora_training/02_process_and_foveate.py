"""
Step 2: Process Bridge TFDS dataset + apply foveated preprocessing offline.

For each episode:
  1. Extract raw images (image_0) + actions from bridge_orig TFDS
  2. Run DINO on first frame to get fovea center (cached per episode)
  3. Apply foveated_tokenize + reconstruct_foveated_with_context to every frame
  4. Save foveated JPEGs + actions.npy + instruction.txt

Output layout:
  output_dir/
    00001/
      images/          ← foveated JPEGs (frame_0001.jpg ...)
      actions/
        actions.npy    ← normalized 7-dim actions  (N, 7)
      instruction.txt
    00002/
      ...

Usage:
    python 02_process_and_foveate.py \
        --dataset-dir  /local-scratch/bridge_orig/0.1.0 \
        --output-dir   /local-scratch/bridge_foveated \
        --vq-hub       /content/pretrain/Emu3-VisionTokenizer \
        --max-episodes 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

# ── Repo path setup ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / ".." / ".." / "..").resolve()
_EXP  = (_HERE / "..").resolve()
for _p in [str(_ROOT), str(_EXP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.foveated_tokenization.foveated_tokenization import (  # noqa: E402
    foveated_tokenize,
    reconstruct_foveated_with_context,
)
from experiments.foveated_tokenization.grounding_dino_wrapper import GroundingDINOWrapper  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only for TFDS processing


# ── Bridge action utilities (from tools/utils.py) ─────────────────────────────

def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Convert continuous gripper actions to binary {0, 1}."""
    open_mask  = actions > 0.95
    close_mask = actions < 0.05
    in_between = ~(open_mask | close_mask)

    is_open_float  = tf.cast(open_mask,  tf.float32)
    is_close_float = tf.cast(close_mask, tf.float32)
    result = tf.where(in_between, tf.zeros_like(actions), is_open_float)
    result = tf.where(close_mask, -tf.ones_like(actions), result)
    return result


def relabel_bridge_actions(traj: dict) -> dict:
    """Relabel bridge gripper actions to [-1, 1] convention."""
    gripper_actions = traj["action"][:, -1]
    binarized = binarize_gripper_actions(gripper_actions)
    # shift by 1 step (next-step gripper state)
    new_gripper = tf.concat([binarized[1:], binarized[-1:]], axis=0)
    traj["action"] = tf.concat(
        [traj["action"][:, :6], new_gripper[:, None]], axis=-1
    )
    return traj


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Normalize actions to [-1, 1] using Bridge dataset quantile stats."""
    q01 = np.array([-0.02887803485105428, -0.04178320091122349, -0.026113155505000457,
                    -0.08117201867235568, -0.09309056401752747, -0.20778717060421048, -1e-10])
    q99 = np.array([ 0.02819482065335177,  0.04079562563528227,  0.04015785568112329,
                     0.08070396399877966,  0.07745134926258301,  0.2016542930635028, 0.99980000009996])
    normalized = 2 * (actions - q01) / (q99 - q01 + 1e-8) - 1
    return np.clip(normalized, -1.0, 1.0)


# ── Main processing ────────────────────────────────────────────────────────────

def process_dataset(
    dataset_dir: str,
    output_dir: str,
    dino_model: str,
    dino_threshold: float,
    blur_scale: float,
    max_episodes: int,
    device: str,
) -> None:
    print(f"[bridge] Loading TFDS from {dataset_dir} ...")
    builder = tfds.builder_from_directory(dataset_dir)
    ds = builder.as_dataset(split="train")

    dino = GroundingDINOWrapper(
        model_name=dino_model,
        box_threshold=dino_threshold,
        text_threshold=dino_threshold,
        device=device,
        cache_steps=1,          # re-detect every step during offline processing
    )

    os.makedirs(output_dir, exist_ok=True)
    skipped = 0
    processed = 0

    for ep_idx, episode in enumerate(tqdm(ds, desc="episodes")):
        if processed >= max_episodes:
            break

        steps = list(episode["steps"])
        if len(steps) < 5:
            skipped += 1
            continue

        # Extract instruction (from first non-empty step)
        instruction = ""
        for step in steps:
            txt = step["language_instruction"].numpy().decode("utf-8").strip()
            if txt:
                instruction = txt
                break
        if not instruction:
            skipped += 1
            continue

        # Extract raw images + actions
        raw_images, actions = [], []
        for i, step in enumerate(steps):
            if i == 0:
                continue  # bridge: skip first step (no action)
            img_np = step["observation"]["image_0"].numpy()  # (H, W, 3)
            act    = step["action"].numpy()                   # (7,)
            raw_images.append(img_np)
            actions.append(act)

        if len(raw_images) < 4:
            skipped += 1
            continue

        # Normalize actions
        actions_np = np.stack(actions, axis=0)               # (N, 7)
        actions_np = relabel_bridge_actions({
            "action": tf.convert_to_tensor(actions_np, dtype=tf.float32)
        })["action"].numpy()
        actions_np = normalize_actions(actions_np)

        # Get fovea center from first frame via DINO
        noun = GroundingDINOWrapper.extract_target_noun(instruction)
        try:
            det = dino.detect(raw_images[0], noun)
        except Exception:
            det = None
        if det is None:
            cx, cy = raw_images[0].shape[1] // 2, raw_images[0].shape[0] // 2
        else:
            cx, cy = int(det[0]), int(det[1])

        # Apply foveated preprocessing to every frame
        ep_dir    = os.path.join(output_dir, f"{ep_idx + 1:05d}")
        img_dir   = os.path.join(ep_dir, "images")
        act_dir   = os.path.join(ep_dir, "actions")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(act_dir, exist_ok=True)

        foveated_paths = []
        for frame_idx, raw_img in enumerate(raw_images):
            patches   = foveated_tokenize(raw_img, cx, cy)
            fov_img   = reconstruct_foveated_with_context(
                raw_img, patches, blur_scale=blur_scale
            )
            out_path = os.path.join(img_dir, f"{frame_idx:04d}.jpg")
            Image.fromarray(fov_img).save(out_path, quality=95)
            foveated_paths.append(out_path)

        np.save(os.path.join(act_dir, "actions.npy"), actions_np)
        with open(os.path.join(ep_dir, "instruction.txt"), "w") as f:
            f.write(instruction)

        processed += 1

    print(f"\n[done] Processed: {processed}  Skipped: {skipped}")
    print(f"       Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Bridge TFDS → foveated images")
    parser.add_argument("--dataset-dir",  required=True,  help="Path to bridge_orig TFDS dir")
    parser.add_argument("--output-dir",   required=True,  help="Output directory")
    parser.add_argument("--dino-model",   default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--dino-threshold", type=float, default=0.15)
    parser.add_argument("--blur-scale",   type=float, default=0.06)
    parser.add_argument("--max-episodes", type=int,   default=5000,
                        help="Max episodes to process (5000 ≈ 30 GB)")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dino_model=args.dino_model,
        dino_threshold=args.dino_threshold,
        blur_scale=args.blur_scale,
        max_episodes=args.max_episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
