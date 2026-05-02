"""
Step 3: Build training pickle from processed foveated Bridge episodes.

Reads output of 02_process_and_foveate.py and produces a pickle file
compatible with Emu3SFTDataset (raw_image=True mode).

Each entry in the pickle:
  {
    "text":          str,            # language instruction
    "image":         [str, ...],     # paths to foveated JPEG frames
    "gripper_image": [],             # unused (no wrist camera in Bridge)
    "action":        np.ndarray,     # (N, 7) normalized actions
  }

Usage:
    python 03_make_pickle.py \
        --processed-dir /local-scratch/bridge_foveated \
        --output-pkl    /local-scratch/bridge_foveated_train.pkl \
        --min-frames    8
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def build_pickle(processed_dir: str, output_pkl: str, min_frames: int) -> None:
    episodes = sorted(Path(processed_dir).iterdir())
    result = []
    skipped = 0

    for ep_dir in tqdm(episodes, desc="building pickle"):
        if not ep_dir.is_dir():
            continue

        instr_file  = ep_dir / "instruction.txt"
        action_file = ep_dir / "actions" / "actions.npy"
        img_dir     = ep_dir / "images"

        if not (instr_file.exists() and action_file.exists() and img_dir.exists()):
            skipped += 1
            continue

        instruction = instr_file.read_text().strip()
        actions     = np.load(str(action_file))          # (N, 7)
        img_paths   = sorted(img_dir.glob("*.jpg"))

        if len(img_paths) < min_frames or len(img_paths) != len(actions):
            skipped += 1
            continue

        result.append({
            "text":          instruction,
            "image":         [str(p) for p in img_paths],
            "gripper_image": [],
            "action":        actions,
        })

    print(f"\n[pickle] Valid episodes : {len(result)}")
    print(f"[pickle] Skipped        : {skipped}")

    os.makedirs(os.path.dirname(os.path.abspath(output_pkl)), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"[pickle] Saved → {output_pkl}")


def main():
    parser = argparse.ArgumentParser(description="Build training pickle")
    parser.add_argument("--processed-dir", required=True,
                        help="Output of 02_process_and_foveate.py")
    parser.add_argument("--output-pkl", required=True,
                        help="Output pickle file path")
    parser.add_argument("--min-frames", type=int, default=8,
                        help="Minimum frames per episode")
    args = parser.parse_args()

    build_pickle(args.processed_dir, args.output_pkl, args.min_frames)


if __name__ == "__main__":
    main()
