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
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        if len(results["boxes"]) == 0:
            return None

        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].cpu().numpy()  # (x_min, y_min, x_max, y_max)
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        return cx, cy

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

    def reset(self) -> None:
        """Reset the step cache (call at the start of each episode)."""
        self._cache_cx = None
        self._cache_cy = None
        self._cache_step = 0
