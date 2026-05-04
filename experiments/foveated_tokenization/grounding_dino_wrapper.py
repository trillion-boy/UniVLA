"""
Grounding DINO wrapper for fovea center detection.

Given an RGB image and a task instruction, returns the (cx, cy) pixel
coordinate of the primary manipulation target to use as the fovea center.

Uses transformers AutoModelForZeroShotObjectDetection (Grounding DINO).
Falls back to image center when detection confidence is too low.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


class GroundingDINOWrapper:
    """
    Lightweight wrapper around Grounding DINO for fovea center detection.

    Usage:
        dino = GroundingDINOWrapper()
        cx, cy = dino.get_fovea_center(image_rgb, "put eggplant in basket")
    """

    # Regex patterns to extract the primary object from manipulation instructions
    _VERB_PATTERNS = [
        r"(?:pick up|grasp|grab|lift|take)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:put|place|move|transfer)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:in|on|into|onto|to)",
        r"(?:push|slide|pull)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:open|close)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:stack)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
    ]

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
        cache_steps: int = 5,
        debug_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_steps = cache_steps

        self._model = None
        self._processor = None

        # N-step cache
        self._cache_cx: Optional[float] = None
        self._cache_cy: Optional[float] = None
        self._cache_step: int = 0

        # Debug visualization
        self._debug_dir = debug_dir
        self._debug_count = 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[DINO] Loading {self.model_name} ...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        print("[DINO] Model loaded.")

    @staticmethod
    def extract_target_noun(instruction: str) -> str:
        """Extract the primary manipulation target noun from a task instruction."""
        instr = instruction.lower().strip()
        for pat in GroundingDINOWrapper._VERB_PATTERNS:
            m = re.search(pat, instr)
            if m:
                return m.group(1).strip()
        # Fallback: pass the whole instruction; DINO handles open-ended text
        return instr

    def detect(
        self,
        image_rgb: np.ndarray,
        text_query: str,
    ) -> Optional[Tuple[float, float]]:
        """
        Run Grounding DINO and return (cx, cy) of the highest-confidence box.
        Returns None if no detection passes the thresholds.
        """
        self._load_model()

        pil_img = Image.fromarray(image_rgb)
        # DINO expects the text query to end with a period
        if not text_query.endswith("."):
            text_query = text_query + "."

        inputs = self._processor(
            images=pil_img,
            text=text_query,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])  # (H, W)
        # First pass with threshold=0.0 to see raw best score for debugging
        try:
            all_results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=0.0, text_threshold=0.0,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            all_results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=0.0, target_sizes=target_sizes,
            )[0]

        if len(all_results["boxes"]) > 0:
            best_raw = all_results["scores"].max().item()
            print(f"[DINO] Best raw score={best_raw:.3f} (threshold={self.box_threshold})")
        else:
            print("[DINO] No boxes at all from model.")
            return None

        # Second pass with actual threshold
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]

        if len(results["boxes"]) == 0:
            return None

        best_idx = results["scores"].argmax().item()
        score = results["scores"][best_idx].item()
        box = results["boxes"][best_idx].cpu().numpy()  # (x_min, y_min, x_max, y_max)
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)

        # Save debug visualization if requested
        if self._debug_dir is not None:
            self._save_debug_image(image_rgb, box, cx, cy, score, text_query)

        return cx, cy

    def detect_bbox(
        self,
        image_rgb: np.ndarray,
        text_query: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Run Grounding DINO and return (x1, y1, x2, y2) of highest-confidence box.
        Returns None if no detection passes the thresholds.
        """
        self._load_model()

        pil_img = Image.fromarray(image_rgb)
        if not text_query.endswith("."):
            text_query = text_query + "."

        inputs = self._processor(
            images=pil_img, text=text_query, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]

        if len(results["boxes"]) == 0:
            return None

        best_idx = results["scores"].argmax().item()
        score    = results["scores"][best_idx].item()
        box      = results["boxes"][best_idx].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        print(f"[DINO] BBox '{text_query.rstrip('.')}' score={score:.3f} "
              f"→ [{x1},{y1},{x2},{y2}]")
        return x1, y1, x2, y2

    def _save_debug_image(
        self,
        image_rgb: np.ndarray,
        box: np.ndarray,
        cx: float,
        cy: float,
        score: float,
        label: str,
    ) -> None:
        import os
        from PIL import ImageDraw, ImageFont
        os.makedirs(self._debug_dir, exist_ok=True)
        img = Image.fromarray(image_rgb).copy()
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(0, 255, 0))
        draw.text((x0, max(0, y0 - 14)), f"{label} {score:.2f}", fill=(255, 0, 0))
        fname = os.path.join(self._debug_dir, f"dino_det_{self._debug_count:04d}.png")
        img.save(fname)
        self._debug_count += 1
        print(f"[DINO] Debug image saved: {fname}")

    def get_fovea_center(
        self,
        image_rgb: np.ndarray,
        instruction: str,
    ) -> Tuple[int, int]:
        """
        Return fovea center (cx, cy) in pixel coordinates.

        Caches the result for `cache_steps` steps to avoid running DINO
        at every inference step. Falls back to image center on failure.
        """
        H, W = image_rgb.shape[:2]

        # Cache hit: skip DINO
        if (
            self._cache_cx is not None
            and self._cache_step % self.cache_steps != 0
        ):
            self._cache_step += 1
            return int(self._cache_cx), int(self._cache_cy)

        noun = self.extract_target_noun(instruction)
        try:
            result = self.detect(image_rgb, noun)
        except Exception as e:
            print(f"[DINO] Detection error: {e}. Falling back to image center.")
            result = None

        if result is None:
            cx, cy = W // 2, H // 2
            print(f"[DINO] No detection for '{noun}', using image center ({cx},{cy}).")
        else:
            cx, cy = int(result[0]), int(result[1])
            print(f"[DINO] Detected '{noun}' at ({cx},{cy}).")

        self._cache_cx = cx
        self._cache_cy = cy
        self._cache_step += 1
        return cx, cy

    @staticmethod
    def extract_source_dest_nouns(instruction: str) -> Tuple[str, Optional[str]]:
        """
        Parse instruction into (source_noun, dest_noun).
        Returns dest_noun=None when no clear destination is found.
        Examples:
            "stack the green block on the yellow block" → ("green block", "yellow block")
            "put the eggplant in the basket"            → ("eggplant", "basket")
            "pick up the spoon"                         → ("spoon", None)
        """
        instr = instruction.lower().strip()

        # Patterns that capture both source and destination
        biobj_patterns = [
            r"(?:stack|put|place|move|transfer)\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto|to)\s+(?:the\s+)?(.+?)(?:\s*$|\s+and\s)",
        ]
        for pat in biobj_patterns:
            m = re.search(pat, instr)
            if m:
                src = m.group(1).strip().rstrip(".,")
                dst = m.group(2).strip().rstrip(".,")
                return src, dst

        # Fallback: only source
        src = GroundingDINOWrapper.extract_target_noun(instruction)
        return src, None

    def get_dual_object_center(
        self,
        image_rgb: np.ndarray,
        instruction: str,
    ) -> Tuple[int, int]:
        """
        Detect source AND destination objects; return their pixel midpoint.
        Falls back to single-object center when dest is not parseable or not detected.
        Cached for cache_steps like get_fovea_center.
        """
        H, W = image_rgb.shape[:2]

        if (
            self._cache_cx is not None
            and self._cache_step % self.cache_steps != 0
        ):
            self._cache_step += 1
            return int(self._cache_cx), int(self._cache_cy)

        src_noun, dst_noun = self.extract_source_dest_nouns(instruction)

        try:
            src_pt = self.detect(image_rgb, src_noun)
        except Exception as e:
            print(f"[DINO] src detection error: {e}")
            src_pt = None

        dst_pt = None
        if dst_noun:
            try:
                dst_pt = self.detect(image_rgb, dst_noun)
            except Exception as e:
                print(f"[DINO] dst detection error: {e}")

        if src_pt is not None and dst_pt is not None:
            cx = int((src_pt[0] + dst_pt[0]) / 2)
            cy = int((src_pt[1] + dst_pt[1]) / 2)
            print(f"[DINO] dual obj: src='{src_noun}' @ {src_pt}, dst='{dst_noun}' @ {dst_pt} → mid=({cx},{cy})")
        elif src_pt is not None:
            cx, cy = int(src_pt[0]), int(src_pt[1])
            print(f"[DINO] single obj: src='{src_noun}' @ ({cx},{cy}), dst not detected")
        else:
            cx, cy = W // 2, H // 2
            print(f"[DINO] no detection for '{src_noun}', using image center ({cx},{cy})")

        self._cache_cx = cx
        self._cache_cy = cy
        self._cache_step += 1
        return cx, cy

    def reset(self) -> None:
        """Reset the step cache (call at the start of each episode)."""
        self._cache_cx = None
        self._cache_cy = None
        self._cache_step = 0
