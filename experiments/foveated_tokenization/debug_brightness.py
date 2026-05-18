#!/usr/bin/env python3
"""
debug_brightness.py

Visually verify that brightness augmentation is working correctly.
Creates a side-by-side comparison image: original vs. darkened versions.

Usage (no GPU, no model needed):
  python debug_brightness.py --image <path_to_any_image.png> --output /tmp/brightness_check.png
  python debug_brightness.py --from-gif <path.gif> --output /tmp/brightness_check.png

Or without any image (generates a synthetic test pattern):
  python debug_brightness.py --output /tmp/brightness_check.png
"""
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import os


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    pil = Image.fromarray(image)
    return np.array(ImageEnhance.Brightness(pil).enhance(factor))


def make_label(text: str, width: int, height: int = 30) -> np.ndarray:
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, fill=(0, 0, 0))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    default=None, help="Path to input image (PNG/JPG)")
    parser.add_argument("--from-gif", default=None, help="Path to GIF — uses first frame")
    parser.add_argument("--output",   default="/tmp/brightness_check.png")
    args = parser.parse_args()

    # ── Load or synthesize a source image ──────────────────────────────────
    if args.from_gif:
        gif = Image.open(args.from_gif)
        gif.seek(0)
        src = np.array(gif.convert("RGB"))
        print(f"[OK] Loaded first frame from GIF: {args.from_gif}  shape={src.shape}")
    elif args.image:
        src = np.array(Image.open(args.image).convert("RGB"))
        print(f"[OK] Loaded image: {args.image}  shape={src.shape}")
    else:
        # Synthetic test pattern: gradient + checkerboard
        h, w = 240, 320
        src = np.zeros((h, w, 3), dtype=np.uint8)
        # Horizontal gradient (R channel)
        src[:, :, 0] = np.tile(np.linspace(50, 230, w, dtype=np.uint8), (h, 1))
        # Vertical gradient (G channel)
        src[:, :, 1] = np.tile(np.linspace(50, 230, h, dtype=np.uint8)[:, None], (1, w))
        # Checkerboard (B channel)
        xg, yg = np.meshgrid(np.arange(w), np.arange(h))
        src[:, :, 2] = ((xg // 20 + yg // 20) % 2) * 200 + 30
        print(f"[OK] Using synthetic test pattern  shape={src.shape}")

    factors = [1.0, 0.8, 0.6, 0.5]
    labels  = ["1.0 (original)", "0.8 (dark)", "0.6 (darker)", "0.5 (darkest)"]

    h, w = src.shape[:2]
    label_h = 30
    padding = 4

    # Build side-by-side comparison
    cols = []
    for factor, label in zip(factors, labels):
        img = apply_brightness(src, factor)
        lbl = make_label(f"brightness={label}", w, label_h)
        col = np.vstack([lbl, img])
        cols.append(col)

    # Add separator lines
    sep = np.full((h + label_h, padding, 3), 200, dtype=np.uint8)
    combined = cols[0]
    for col in cols[1:]:
        combined = np.hstack([combined, sep, col])

    out = Image.fromarray(combined)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.save(args.output)
    print(f"\n[Saved] {args.output}  ({out.width}×{out.height})")

    # Print pixel statistics per factor
    print("\n── Mean pixel brightness ──────────────────────────")
    for factor, label in zip(factors, labels):
        img = apply_brightness(src, factor)
        mean = img.mean()
        print(f"  factor={factor}:  mean={mean:.1f}  (original mean={src.mean():.1f})")
    print()


if __name__ == "__main__":
    main()
