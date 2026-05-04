"""
Step 2 (token-level LoRA): Extract clean Bridge frames + DINO fovea centers.

Unlike 02_process_and_foveate.py, this does NOT apply any image-level blur.
VQ-VAE will always see clean pixels (zero distribution shift).
Fovea center is stored per-episode for use during LoRA training.

Output layout:
  output_dir/
    00001/
      images/          ← clean JPEGs (frame_0000.jpg ...)
      actions/
        actions.npy    ← normalized 7-dim actions  (N, 7)
      instruction.txt
      fovea_center.txt ← "cx_norm cy_norm" (normalized 0-1)
    00002/
      ...

Usage:
    python 02_process_clean.py \
        --dataset-dir  /content/bridge_orig/0.1.0 \
        --output-dir   /content/bridge_clean \
        --max-episodes 3000
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

_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / ".." / ".." / "..").resolve()
_EXP  = (_HERE / "..").resolve()
for _p in [str(_ROOT), str(_EXP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.foveated_tokenization.grounding_dino_wrapper import GroundingDINOWrapper  # noqa

tf.config.set_visible_devices([], "GPU")


def binarize_gripper_actions(actions):
    open_mask  = actions > 0.95
    close_mask = actions < 0.05
    in_between = ~(open_mask | close_mask)
    result = tf.where(in_between, tf.zeros_like(actions),
                      tf.cast(open_mask, tf.float32))
    result = tf.where(close_mask, -tf.ones_like(actions), result)
    return result


def relabel_bridge_actions(traj):
    gripper = binarize_gripper_actions(traj["action"][:, -1])
    new_g   = tf.concat([gripper[1:], gripper[-1:]], axis=0)
    traj["action"] = tf.concat([traj["action"][:, :6], new_g[:, None]], axis=-1)
    return traj


def normalize_actions(actions):
    q01 = np.array([-0.02887803485105428, -0.04178320091122349, -0.026113155505000457,
                    -0.08117201867235568, -0.09309056401752747, -0.20778717060421048, -1e-10])
    q99 = np.array([ 0.02819482065335177,  0.04079562563528227,  0.04015785568112329,
                     0.08070396399877966,  0.07745134926258301,  0.2016542930635028,  0.9998])
    return np.clip(2 * (actions - q01) / (q99 - q01 + 1e-8) - 1, -1.0, 1.0)


def process_dataset(dataset_dir, output_dir, max_episodes, device):
    print(f"[bridge] Loading TFDS from {dataset_dir} ...")
    builder = tfds.builder_from_directory(dataset_dir)
    ds = builder.as_dataset(split="train")

    dino = GroundingDINOWrapper(
        model_name="IDEA-Research/grounding-dino-tiny",
        box_threshold=0.15, text_threshold=0.15,
        device=device, cache_steps=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    processed = skipped = 0

    for ep_idx, episode in enumerate(tqdm(ds, desc="episodes")):
        if processed >= max_episodes:
            break

        steps = list(episode["steps"])
        if len(steps) < 5:
            skipped += 1; continue

        instruction = ""
        for step in steps:
            txt = step["language_instruction"].numpy().decode("utf-8").strip()
            if txt:
                instruction = txt; break
        if not instruction:
            skipped += 1; continue

        raw_images, actions = [], []
        for i, step in enumerate(steps):
            if i == 0:
                continue
            img_np = step["observation"]["image_0"].numpy()
            act    = step["action"].numpy()
            raw_images.append(img_np)
            actions.append(act)

        if len(raw_images) < 4:
            skipped += 1; continue

        actions_np = normalize_actions(
            relabel_bridge_actions({
                "action": tf.convert_to_tensor(np.stack(actions), dtype=tf.float32)
            })["action"].numpy()
        )

        # DINO fovea center on first frame
        noun = GroundingDINOWrapper.extract_target_noun(instruction)
        H, W = raw_images[0].shape[:2]
        try:
            det = dino.detect(raw_images[0], noun)
        except Exception:
            det = None
        if det is None:
            cx_norm, cy_norm = 0.5, 0.5
        else:
            cx_norm = float(det[0]) / W
            cy_norm = float(det[1]) / H

        ep_dir  = Path(output_dir) / f"{ep_idx + 1:05d}"
        img_dir = ep_dir / "images"
        act_dir = ep_dir / "actions"
        img_dir.mkdir(parents=True, exist_ok=True)
        act_dir.mkdir(parents=True, exist_ok=True)

        for fi, raw_img in enumerate(raw_images):
            Image.fromarray(raw_img).save(img_dir / f"{fi:04d}.jpg", quality=95)

        np.save(act_dir / "actions.npy", actions_np)
        (ep_dir / "instruction.txt").write_text(instruction)
        (ep_dir / "fovea_center.txt").write_text(f"{cx_norm:.6f} {cy_norm:.6f}")
        processed += 1

    print(f"\n[done] Processed={processed}  Skipped={skipped}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir",  required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--max-episodes", type=int, default=3000)
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()
    process_dataset(args.dataset_dir, args.output_dir,
                    args.max_episodes, args.device)


if __name__ == "__main__":
    main()
