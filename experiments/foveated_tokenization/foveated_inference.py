"""
FoveatedEmuVLAInference — Path 1 implementation.

Extends EmuVLAInference by overriding preprocess() to:
  1. Detect fovea center from the task instruction via Grounding DINO
  2. Apply foveated tokenization + bio-inspired context reconstruction
     (sharp center + blurry periphery → same resolution as original)
  3. Feed the reconstructed image through the existing VQ-VAE unchanged

The VQ-VAE, LLM, and action tokenizer are identical to the baseline —
only the input image distribution changes.

Paths:
  emu_hub, vq_hub, vision_hub — same as EmuVLAInference
  fast_path                   — base directory containing fast_bridge_t5_s50 /
                                fast_google_a5_s50 subdirs (overrides the
                                hardcoded /share/project/... path in parent)
"""

from __future__ import annotations

import os
import sys
from queue import Queue
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── Repo path setup ────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROBOVLMS = os.path.join(_ROOT, "reference", "RoboVLMs")
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
for _p in [_ROBOVLMS, _EMU3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.simpler.model_wrapper import EmuVLAInference  # noqa: E402
from experiments.foveated_tokenization.foveated_tokenization import (  # noqa: E402
    foveated_tokenize,
    reconstruct_foveated_with_context,
)
from experiments.foveated_tokenization.grounding_dino_wrapper import (  # noqa: E402
    GroundingDINOWrapper,
)


class FoveatedEmuVLAInference(EmuVLAInference):
    """
    Drop-in replacement for EmuVLAInference with foveated preprocessing.

    Example (Colab):
        model = FoveatedEmuVLAInference(
            emu_hub="/content/pretrain/UniVLA",
            vq_hub="/content/pretrain/Emu3-VisionTokenizer",
            vision_hub="/content/pretrain/Emu3-VisionTokenizer",
            device="cuda",
            policy_setup="widowx_bridge",
            fast_path="/content/pretrain",
            dino_cache_steps=5,
            blur_scale=0.06,
        )
        raw_actions, env_actions = model.step(image_np, "put eggplant in basket")
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        # Path override for hardcoded fast-tokenizer location in parent
        fast_path: Optional[str] = None,
        # Foveated-specific parameters
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        blur_scale: float = 0.06,
    ):
        # Store before super().__init__() so init_config() override can read them
        self._fast_path_override = fast_path
        self.blur_scale = blur_scale
        self._current_instruction: str = ""

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
        )

        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    # ── Override init_config to fix hardcoded fast-tokenizer paths ─────────────
    def init_config(self, device: str) -> None:
        """Identical to parent but uses self._fast_path_override for fast tokenizer."""
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
        )
        from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer

        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vision_hub, trust_remote_code=True
        )
        self.image_tokenizer = (
            AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            .to(device)
            .eval()
        )
        self.processor = Emu3Processor(
            self.image_processor, self.image_tokenizer, self.tokenizer
        )

        # Fast tokenizer path (configurable)
        base = self._fast_path_override or "/share/project/yuqi.wang/UniVLA/pretrain"
        if self.policy_setup == "widowx_bridge":
            fast_path = os.path.join(base, "fast_bridge_t5_s50")
        elif self.policy_setup == "google_robot":
            fast_path = os.path.join(base, "fast_google_a5_s50")
        else:
            fast_path = os.path.join(base, "fast")

        self.action_tokenizer = AutoProcessor.from_pretrained(
            fast_path, trust_remote_code=True
        )

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    # ── Foveated preprocess ────────────────────────────────────────────────────
    def preprocess(self, image: np.ndarray) -> Tuple:
        """
        Foveated preprocessing pipeline:
          1. Detect fovea center from DINO (cached every N steps; fallback = image center)
          2. Foveated tokenize + reconstruct (sharp center, blurry periphery)
          3. VQ-VAE encode — identical to baseline preprocess()
        """
        H, W = image.shape[:2]

        if self._current_instruction:
            cx, cy = self.dino.get_fovea_center(image, self._current_instruction)
        else:
            cx, cy = W // 2, H // 2

        patches = foveated_tokenize(image, cx, cy)
        fov_image = reconstruct_foveated_with_context(
            image, patches, blur_scale=self.blur_scale
        )

        agent_view = Image.fromarray(fov_image).resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        return image_code, None

    # ── Override step to keep instruction in sync ──────────────────────────────
    def step(self, image: np.ndarray, goal: str):
        self._current_instruction = goal
        return super().step(image, goal)

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""
