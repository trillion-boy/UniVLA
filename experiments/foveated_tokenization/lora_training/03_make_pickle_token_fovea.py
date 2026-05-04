"""
Step 3 (token-level LoRA): Build training pickle from clean Bridge episodes.

Same format as 03_make_pickle.py but adds 'fovea_center' (cx_norm, cy_norm)
per episode so the trainer can apply the circular token mask.

Usage:
    python 03_make_pickle_token_fovea.py \
        --processed-dir /content/bridge_clean \
        --output-pkl    /content/bridge_token_fovea_train.pkl \
        --min-frames    8
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def build_pickle(processed_dir, output_pkl, min_frames):
    episodes = sorted(Path(processed_dir).iterdir())
    result = []
    skipped = 0

    for ep_dir in tqdm(episodes, desc="building pickle"):
        if not ep_dir.is_dir():
            continue

        instr_file  = ep_dir / "instruction.txt"
        action_file = ep_dir / "actions" / "actions.npy"
        img_dir     = ep_dir / "images"
        fovea_file  = ep_dir / "fovea_center.txt"

        if not (instr_file.exists() and action_file.exists() and img_dir.exists()):
            skipped += 1; continue

        instruction = instr_file.read_text().strip()
        actions     = np.load(str(action_file))
        img_paths   = sorted(img_dir.glob("*.jpg"))

        if len(img_paths) < min_frames or len(img_paths) != len(actions):
            skipped += 1; continue

        # Load fovea center (default to image center if missing)
        if fovea_file.exists():
            cx_norm, cy_norm = map(float, fovea_file.read_text().split())
        else:
            cx_norm, cy_norm = 0.5, 0.5

        result.append({
            "text":          instruction,
            "image":         [str(p) for p in img_paths],
            "gripper_image": [],
            "action":        actions,
            "fovea_center":  (cx_norm, cy_norm),   # ← NEW: used for token mask
        })

    print(f"\n[pickle] Valid: {len(result)}  Skipped: {skipped}")
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"[pickle] Saved → {output_pkl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--output-pkl",    required=True)
    p.add_argument("--min-frames",    type=int, default=8)
    args = p.parse_args()
    build_pickle(args.processed_dir, args.output_pkl, args.min_frames)


if __name__ == "__main__":
    main()
