"""
Standalone foveated inference module.

Contains:
  EmuVLAInference       — copied from reference/RoboVLMs/eval/simpler/model_wrapper.py
                          with robovlms / lightning dependencies removed.
  FoveatedEmuVLAInference — Path 1 implementation: overrides preprocess() to apply
                          DINO-based foveated reconstruction before VQ-VAE encoding.

Why standalone: model_wrapper.py pulls in eval.calvin.model_wrapper → eval_utils →
lightning.pytorch.loggers, which creates a deep dependency chain that isn't needed
for EmuVLAInference inference. Copying the class here breaks that chain.
"""

from __future__ import annotations

import os
import sys
from queue import Queue
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessor,
)
from transformers.feature_extraction_utils import BatchFeature

# ── Repo path setup ────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
# _ROOT must be in sys.path so emu3/modeling_emu3.py can find `models.policy_head`
for _p in [_ROOT, _EMU3]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer  # noqa: E402

# ── Foveated helpers ───────────────────────────────────────────────────────────
_EXP_DIR = os.path.dirname(__file__)
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from foveated_tokenization import (  # noqa: E402
    foveated_tokenize,
    reconstruct_foveated_with_context,
)
from grounding_dino_wrapper import GroundingDINOWrapper  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# ActionIDConstraintLogitsProcessor (copied from model_wrapper.py)
# ══════════════════════════════════════════════════════════════════════════════

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


# ══════════════════════════════════════════════════════════════════════════════
# EmuVLAInference  (copied from model_wrapper.py, CustomModel dependency removed)
# ══════════════════════════════════════════════════════════════════════════════

class EmuVLAInference:
    """
    Baseline UniVLA inference wrapper for SimplerEnv.
    Copied verbatim from reference/RoboVLMs/eval/simpler/model_wrapper.py
    with the CustomModel base class removed (EmuVLAInference overrides
    every method anyway, so the base class is unused).
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
    ):
        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.policy_setup = policy_setup
        # Only set if not already set by a subclass __init__ (called before super())
        if not hasattr(self, "_fast_path_override"):
            self._fast_path_override = fast_path

        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
            self.image_size = (160, 128)
        elif self.policy_setup == "widowx_bridge":
            self.close_gripper_act = 1
            self.image_size = (256, 256)

        self.sticky_gripper_num_repeat = 2
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.close_gripper_num = 0
        self.late_close_gripper = 1

        self.window_size = 2
        self.predict_action_frames = 5
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = False
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.use_cot = False
        self.video_mode = True

        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

        self.kwargs = dict(mode="VLA_COT" if self.use_cot else "VLA", padding="longest")

        if self.use_fast:
            self.GENERATION_CONFIG = GenerationConfig(
                pad_token_id=self.model.config.pad_token_id,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.eoa_token_id,
                do_sample=False,
            )
        else:
            self.GENERATION_CONFIG = GenerationConfig(
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_new_tokens=800,
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )

    def init_config(self, device: str) -> None:
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.emu_hub,
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

        # fast tokenization path — use override if provided, else try default location
        _fast_base = getattr(self, "_fast_path_override", None) or \
            "/share/project/yuqi.wang/UniVLA/pretrain"
        if self.policy_setup == "widowx_bridge":
            fast_path = os.path.join(_fast_base, "fast_bridge_t5_s50")
        elif self.policy_setup == "google_robot":
            fast_path = os.path.join(_fast_base, "fast_google_a5_s50")
        else:
            fast_path = os.path.join(_fast_base, "fast")
        if os.path.isdir(fast_path):
            self.action_tokenizer = AutoProcessor.from_pretrained(
                fast_path, trust_remote_code=True
            )
        else:
            self.action_tokenizer = None  # caller must set this before step()

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)

    def get_history(self):
        return list(self.vision_queue.queue)

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)

    def get_action_history(self):
        return list(self.action_queue.queue)

    def reset(self) -> None:
        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.close_gripper_num = 0
        self.previous_gripper_action = None
        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image: np.ndarray):
        agent_view = Image.fromarray(image).resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        image_code = self.image_tokenizer.encode(image_x)
        return image_code, None

    def step(self, image: np.ndarray, goal: str):
        image_code, gripper_code = self.preprocess(image)
        prompt = goal

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type="pt")

        kwargs = dict(mode="VLA_Video", padding="longest")
        pos_inputs = self.processor.video_process(
            text=prompt,
            video_tokens=video_code,
            gripper_tokens=gripper_code,
            context_frames=self.context_frames,
            frames=self.predict_frames,
            return_tensors="pt",
            **kwargs,
        )

        if self.video_mode:
            self.add_image(pos_inputs)
            history = self.get_history()
            action_history = self.get_action_history()

            all_input_ids = [text_tokens["input_ids"]]
            all_token_type_ids = [text_tokens["token_type_ids"]]
            all_attention_mask = [text_tokens["attention_mask"]]

            for i, hist in enumerate(history):
                if i < len(action_history):
                    act = action_history[i]
                    all_input_ids.extend([hist["input_ids"], act])
                    all_token_type_ids.extend([
                        hist["token_type_ids"],
                        torch.zeros_like(act),
                    ])
                    all_attention_mask.extend([
                        hist["attention_mask"],
                        torch.ones_like(act),
                    ])
                else:
                    all_input_ids.append(hist["input_ids"])
                    all_token_type_ids.append(hist["token_type_ids"])
                    all_attention_mask.append(hist["attention_mask"])

            final_inputs = pos_inputs.copy()
            final_inputs["input_ids"] = torch.cat(all_input_ids, dim=1)
            final_inputs["token_type_ids"] = torch.cat(all_token_type_ids, dim=1)
            final_inputs["attention_mask"] = torch.cat(all_attention_mask, dim=1)
        else:
            final_inputs = pos_inputs

        if self.use_fast:
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed = list(
                range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)
            ) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed)

            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=100,
                    logits_processor=[action_id_processor],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )

            orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            outputs = orig_outputs[:, :-1]
            last_token_id_t = torch.tensor(
                last_token_id, dtype=outputs.dtype, device=outputs.device
            )
            processed = last_token_id_t - outputs
            action_outputs = self.action_tokenizer.decode(
                processed,
                time_horizon=self.predict_action_frames,
                action_dim=self.action_dim,
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        action = self.unormalize_action(action)
        action_pred = action[0:1] if self.use_one_step else action
        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        return [r[0] for r in res], [r[1] for r in res]

    def unormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.policy_setup == "google_robot":
            high = np.array([0.17753939667156038, 0.14669284061245857, 0.2179850059450077,
                             0.5882710857765758, 0.35334834816471683, 0.4470693223284772, 0.99980000009996])
            low  = np.array([-0.22624227067544056, -0.15126218201085617, -0.23251856873127252,
                              -0.3538952427136002, -0.4202906595250484, -0.43766197340888247, -1e-10])
        else:
            high = np.array([0.02819482065335177, 0.04079562563528227, 0.04015785568112329,
                             0.08070396399877966, 0.07745134926258301, 0.2016542930635028, 0.99980000009996])
            low  = np.array([-0.02887803485105428, -0.04178320091122349, -0.026113155505000457,
                              -0.08117201867235568, -0.09309056401752747, -0.20778717060421048, -1e-10])
        return 0.5 * (action + 1) * (high - low) + low

    def transform_action(self, raw_actions: np.ndarray):
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"]
        roll, pitch, yaw = raw_action["rotation_delta"]
        ax, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = ax * angle

        if self.policy_setup == "google_robot":
            cur = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            rel = np.array([0]) if self.previous_gripper_action is None else self.previous_gripper_action - cur
            self.previous_gripper_action = cur
            if np.abs(rel) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        elif self.policy_setup == "widowx_bridge":
            rel = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if rel[0] > 0:
                self.close_gripper_num += 1
            else:
                self.close_gripper_num = 0
            rel[0] = 1 if self.close_gripper_num >= self.late_close_gripper else -1
            action["gripper"] = rel

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action


# ══════════════════════════════════════════════════════════════════════════════
# FoveatedEmuVLAInference  (Path 1)
# ══════════════════════════════════════════════════════════════════════════════

class FoveatedEmuVLAInference(EmuVLAInference):
    """
    Drop-in replacement for EmuVLAInference with foveated preprocessing.

    Overrides:
      init_config() — uses configurable fast_path instead of hardcoded /share/...
      preprocess()  — DINO fovea detection → foveated reconstruction → VQ-VAE
      step()        — keeps _current_instruction in sync with goal
      reset()       — also resets DINO cache
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        blur_scale: float = 0.06,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override = fast_path
        self._lora_path = lora_path
        self.blur_scale = blur_scale
        self._current_instruction: str = ""

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )

        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    def init_config(self, device: str) -> None:
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if getattr(self, "_lora_path", None):
            from peft import PeftModel
            print(f"[lora] Loading LoRA adapter from {self._lora_path} ...")
            self.model = PeftModel.from_pretrained(self.model, self._lora_path)
            self.model = self.model.merge_and_unload()
            print("[lora] Adapter merged into base model.")
        self.model = self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.emu_hub,
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

        _fast_base = self._fast_path_override or "/share/project/yuqi.wang/UniVLA/pretrain"
        if self.policy_setup == "widowx_bridge":
            fast_path = os.path.join(_fast_base, "fast_bridge_t5_s50")
        elif self.policy_setup == "google_robot":
            fast_path = os.path.join(_fast_base, "fast_google_a5_s50")
        else:
            fast_path = os.path.join(_fast_base, "fast")
        if os.path.isdir(fast_path):
            self.action_tokenizer = AutoProcessor.from_pretrained(
                fast_path, trust_remote_code=True
            )
        else:
            self.action_tokenizer = None  # caller must set before step()

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0
        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def preprocess(self, image: np.ndarray):
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

    def step(self, image: np.ndarray, goal: str):
        self._current_instruction = goal
        return super().step(image, goal)

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""


# ══════════════════════════════════════════════════════════════════════════════
# TokenFoveatedEmuVLAInference  (Path 2: token-level foveation)
# ══════════════════════════════════════════════════════════════════════════════

class TokenFoveatedEmuVLAInference(EmuVLAInference):
    """
    Token-level foveation: encode the NORMAL image (no blurring),
    then replace peripheral VQ tokens with a background token.

    Unlike FoveatedEmuVLAInference (image-level blur), this never feeds
    blurry pixels to the VQ-VAE, so there is zero distribution shift.
    The fovea effect is applied AFTER encoding, purely at the token level.

    fovea_fraction: fraction of the token-grid shorter side used as fovea radius.
                    0.4 → circle covering ~50% of image area.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        fovea_fraction: float = 0.4,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override = fast_path
        self._fovea_fraction = fovea_fraction
        self._current_instruction: str = ""

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    def _foveate_tokens(
        self,
        tokens: torch.Tensor,
        cx_px: int, cy_px: int,
        H_img: int, W_img: int,
    ) -> torch.Tensor:
        """Replace tokens outside the fovea circle with the background token."""
        H_t, W_t = tokens.shape[-2], tokens.shape[-1]

        cx_t = cx_px * W_t / W_img
        cy_t = cy_px * H_t / H_img

        y_idx = torch.arange(H_t, dtype=torch.float32)
        x_idx = torch.arange(W_t, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        dist = torch.sqrt((xx - cx_t) ** 2 + (yy - cy_t) ** 2)

        fovea_radius = min(H_t, W_t) * self._fovea_fraction
        peripheral_mask = dist > fovea_radius  # (H_t, W_t) bool

        # Use the most common token in the image as background fill
        bg_token = int(tokens.flatten().mode().values.item())

        tokens_out = tokens.clone()
        tokens_out[0][peripheral_mask] = bg_token
        return tokens_out

    def preprocess(self, image: np.ndarray):
        H, W = image.shape[:2]
        if self._current_instruction:
            cx, cy = self.dino.get_fovea_center(image, self._current_instruction)
        else:
            cx, cy = W // 2, H // 2

        # Encode NORMAL image — VQ-VAE sees clean pixels (no distribution shift)
        agent_view = Image.fromarray(image).resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        image_code = self.image_tokenizer.encode(image_x)  # (1, H_t, W_t)

        # Apply fovea mask at token level
        image_code = self._foveate_tokens(image_code, cx, cy, H, W)
        return image_code, None

    def step(self, image: np.ndarray, goal: str):
        self._current_instruction = goal
        return super().step(image, goal)

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""


# ══════════════════════════════════════════════════════════════════════════════
# TrueFoveatedEmuVLAInference  (Look, Focus, Act style)
# ══════════════════════════════════════════════════════════════════════════════

class TrueFoveatedEmuVLAInference(EmuVLAInference):
    """
    True foveated tokenization (CVPR 2025 "Look, Focus, Act" style).

    Total token count = same as baseline (1024).

    Step 1 — Full image → 1024 peripheral tokens (1 token per 8×8 px patch)
    Step 2 — Center crop (crop_fraction of image) upscaled to full size
              → 1024 tokens, each covering fewer original pixels
              → higher spatial resolution in the fovea region
    Step 3 — Combine:
              inside fovea circle  → center tokens  (high-res)
              outside fovea circle → full-image tokens (low-res context)

    crop_fraction : size of center crop relative to image dimension.
                    0.5 → 128x128 out of 256x256 = 2x resolution boost.
    fovea_fraction: fovea circle radius as fraction of token-grid short side.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        crop_fraction: float = 0.5,
        fovea_fraction: float = 0.35,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override = fast_path
        self._crop_fraction = crop_fraction
        self._fovea_fraction = fovea_fraction
        self._current_instruction: str = ""

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    def preprocess(self, image: np.ndarray):
        H, W = image.shape[:2]
        if self._current_instruction:
            cx, cy = self.dino.get_fovea_center(image, self._current_instruction)
        else:
            cx, cy = W // 2, H // 2

        # ── 1. Full image → peripheral tokens ─────────────────────────────────
        full_view = Image.fromarray(image).resize(self.image_size)
        full_x = self.image_processor(full_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        full_tokens = self.image_tokenizer.encode(full_x)  # (1, H_t, W_t)

        # ── 2. Center crop → upscale → high-res center tokens ─────────────────
        half_h = int(H * self._crop_fraction / 2)
        half_w = int(W * self._crop_fraction / 2)
        x1 = max(0, cx - half_w);  x2 = min(W, cx + half_w)
        y1 = max(0, cy - half_h);  y2 = min(H, cy + half_h)

        crop = image[y1:y2, x1:x2]
        crop_view = Image.fromarray(crop).resize(self.image_size)  # 2x upscale
        crop_x = self.image_processor(crop_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        center_tokens = self.image_tokenizer.encode(crop_x)  # (1, H_t, W_t)

        # ── 3. Combine ─────────────────────────────────────────────────────────
        H_t, W_t = full_tokens.shape[-2], full_tokens.shape[-1]

        cx_t = cx * W_t / W
        cy_t = cy * H_t / H

        y_idx = torch.arange(H_t, dtype=torch.float32)
        x_idx = torch.arange(W_t, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        dist = torch.sqrt((xx - cx_t) ** 2 + (yy - cy_t) ** 2)
        fovea_mask = dist <= min(H_t, W_t) * self._fovea_fraction

        # Map each fovea position in full token grid → corresponding crop token
        ys, xs = torch.where(fovea_mask)
        px = xs.float() * W / W_t   # original pixel x
        py = ys.float() * H / H_t   # original pixel y
        cx_crop = ((px - x1) / (x2 - x1) * W_t).long()
        cy_crop = ((py - y1) / (y2 - y1) * H_t).long()
        valid = (cx_crop >= 0) & (cx_crop < W_t) & (cy_crop >= 0) & (cy_crop < H_t)

        combined = full_tokens.clone()
        combined[0, ys[valid], xs[valid]] = center_tokens[0, cy_crop[valid], cx_crop[valid]]

        return combined, None

    def step(self, image: np.ndarray, goal: str):
        self._current_instruction = goal
        return super().step(image, goal)

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""


# ══════════════════════════════════════════════════════════════════════════════
# TrueFoveatedDualObjEmuVLAInference  (midpoint of source + destination objects)
# ══════════════════════════════════════════════════════════════════════════════

class TrueFoveatedDualObjEmuVLAInference(TrueFoveatedEmuVLAInference):
    """
    Same as TrueFoveatedEmuVLAInference but uses the MIDPOINT of the source
    and destination objects (both detected by DINO) as the fovea center.

    This ensures both manipulation-relevant objects remain inside the crop for
    two-object tasks like "stack green block on yellow block", where centering
    on only the source object pushes the destination outside the crop region.
    """

    def preprocess(self, image: np.ndarray):
        H, W = image.shape[:2]
        if self._current_instruction:
            cx, cy = self.dino.get_dual_object_center(image, self._current_instruction)
        else:
            cx, cy = W // 2, H // 2

        # ── 1. Full image → peripheral tokens ─────────────────────────────────
        full_view = Image.fromarray(image).resize(self.image_size)
        full_x = self.image_processor(full_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        full_tokens = self.image_tokenizer.encode(full_x)  # (1, H_t, W_t)

        # ── 2. Center crop around midpoint → upscale → high-res tokens ────────
        half_h = int(H * self._crop_fraction / 2)
        half_w = int(W * self._crop_fraction / 2)
        x1 = max(0, cx - half_w);  x2 = min(W, cx + half_w)
        y1 = max(0, cy - half_h);  y2 = min(H, cy + half_h)

        crop = image[y1:y2, x1:x2]
        crop_view = Image.fromarray(crop).resize(self.image_size)
        crop_x = self.image_processor(crop_view, return_tensors="pt")[
            "pixel_values"
        ].cuda()
        center_tokens = self.image_tokenizer.encode(crop_x)  # (1, H_t, W_t)

        # ── 3. Combine ─────────────────────────────────────────────────────────
        H_t, W_t = full_tokens.shape[-2], full_tokens.shape[-1]

        cx_t = cx * W_t / W
        cy_t = cy * H_t / H

        y_idx = torch.arange(H_t, dtype=torch.float32)
        x_idx = torch.arange(W_t, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
        dist = torch.sqrt((xx - cx_t) ** 2 + (yy - cy_t) ** 2)
        fovea_mask = dist <= min(H_t, W_t) * self._fovea_fraction

        ys, xs = torch.where(fovea_mask)
        px = xs.float() * W / W_t
        py = ys.float() * H / H_t
        cx_crop = ((px - x1) / (x2 - x1) * W_t).long()
        cy_crop = ((py - y1) / (y2 - y1) * H_t).long()
        valid = (cx_crop >= 0) & (cx_crop < W_t) & (cy_crop >= 0) & (cy_crop < H_t)

        combined = full_tokens.clone()
        combined[0, ys[valid], xs[valid]] = center_tokens[0, cy_crop[valid], cx_crop[valid]]

        return combined, None


# ══════════════════════════════════════════════════════════════════════════════
# BASSEmuVLAInference  (Möbius warp — single coherent warped image to VQ-VAE)
# ══════════════════════════════════════════════════════════════════════════════

class BASSEmuVLAInference(EmuVLAInference):
    """
    Bio-inspired Adaptive Stereographic Sampling (BASS) VLA inference.

    Key difference from TrueFoveated
    ---------------------------------
    TrueFoveated mixes tokens from TWO images (crop + full) → OOD artefact.
    BASS warps ONE image via a Möbius transformation and feeds the warped
    image directly to VQ-VAE → no token mixing, no distribution shift.

    Pipeline per step
    -----------------
    1. DINO detects source object (and destination if two-object task)
    2. DualFocusOptimizer → midpoint pole + reduced strength (both visible)
    3. PhaseAwareMagnification → adjust strength based on gripper state
    4. MobiusWarpModule.warp() → single warped image (pole region magnified)
    5. VQ-VAE encodes warped image → 1024 tokens with higher res at pole
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        grasp_strength: float = 4.0,
        move_strength: float = 2.0,
        dual_focus: bool = True,
        resolution_floor: float = 0.20,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override = fast_path
        self._current_instruction: str = ""
        self._dual_focus = dual_focus

        from bass_warp import MobiusWarpModule, PhaseAwareMagnification, DualFocusOptimizer
        self.warp_module = MobiusWarpModule(resolution_floor=resolution_floor)
        self.phase       = PhaseAwareMagnification(grasp_strength, move_strength)
        self.dual_opt    = DualFocusOptimizer(
            max_strength=grasp_strength,
            min_strength=move_strength,
        )
        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )
        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    def _select_pole(self, image: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """Choose (focus_px, strength) for this frame."""
        H, W = image.shape[:2]
        if not self._current_instruction:
            return (W // 2, H // 2), 1.0

        if self._dual_focus:
            src_noun, dst_noun = GroundingDINOWrapper.extract_source_dest_nouns(
                self._current_instruction
            )
            src_pt, dst_pt = None, None
            try:
                src_pt = self.dino.detect(image, src_noun)
            except Exception:
                pass
            if dst_noun:
                try:
                    dst_pt = self.dino.detect(image, dst_noun)
                except Exception:
                    pass

            if src_pt and dst_pt:
                pole_px, s = self.dual_opt.compute(
                    (int(src_pt[0]), int(src_pt[1])),
                    (int(dst_pt[0]), int(dst_pt[1])),
                    W, H,
                )
                if self.phase.phase == "moving":
                    s = min(s, self.phase.move_strength)
                print(f"[BASS] dual pole={pole_px} s={s:.2f} [{self.phase.phase}]")
                return pole_px, s
            if src_pt:
                pole_px = (int(src_pt[0]), int(src_pt[1]))
                s = self.phase.strength
                print(f"[BASS] single pole={pole_px} s={s:.2f} [{self.phase.phase}]")
                return pole_px, s

        cx, cy = self.dino.get_fovea_center(image, self._current_instruction)
        return (cx, cy), self.phase.strength

    def preprocess(self, image: np.ndarray):
        dev = next(self.model.parameters()).device
        focus_px, strength = self._select_pole(image)

        img_t = (
            torch.from_numpy(image).permute(2, 0, 1)
            .float().div(255.0).unsqueeze(0).to(dev)
        )
        warped = self.warp_module.warp(img_t, focus_px, strength)

        warped_np = (
            warped[0].permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255.0
        ).astype(np.uint8)
        warped_pil = Image.fromarray(warped_np).resize(self.image_size)

        warped_x = self.image_processor(warped_pil, return_tensors="pt")[
            "pixel_values"
        ].to(dev)
        return self.image_tokenizer.encode(warped_x), None

    def step(self, image: np.ndarray, goal: str):
        self._current_instruction = goal
        raw_actions, env_actions = super().step(image, goal)
        if env_actions:
            g = float(np.asarray(env_actions[-1].get("gripper", [1.0])).flat[0])
            self.phase.update((1.0 - g) / 2.0)   # +1=open->0, -1=close->1
        return raw_actions, env_actions

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""
        self.phase.update(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Foveated-Blur utility
# ══════════════════════════════════════════════════════════════════════════════

def apply_foveated_blur(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    blur_ksize: int = 61,
    mask_ksize: int = 41,
    bbox_margin: float = 0.15,
) -> np.ndarray:
    """
    Blend sharp original and heavily blurred version using a soft BBox mask.

    The fovea region (inside DINO bbox) stays pixel-perfect sharp.
    The periphery gets a strong Gaussian blur → only colour / rough shape survives.
    A second Gaussian on the mask itself gives a smooth gradient boundary so
    the VQ-VAE never sees a hard edge discontinuity.

    Args
    ----
    image       : uint8 HWC RGB
    bbox        : (x1, y1, x2, y2) in pixel coordinates
    blur_ksize  : kernel size for background blur (odd, ≥ 3)
    mask_ksize  : kernel size for mask edge softening (odd, ≥ 3)
    bbox_margin : fraction by which to expand bbox before masking

    Returns
    -------
    foveated : uint8 HWC RGB (same shape as input)
    """
    import cv2

    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # ── 1. Expand bbox by margin to avoid clipping object edges ───────────────
    mw = int((x2 - x1) * bbox_margin)
    mh = int((y2 - y1) * bbox_margin)
    x1 = max(0, x1 - mw);  x2 = min(W, x2 + mw)
    y1 = max(0, y1 - mh);  y2 = min(H, y2 + mh)

    # ── 2. Blurred background (strong) ────────────────────────────────────────
    ksize = blur_ksize | 1          # ensure odd
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # ── 3. Hard binary mask: 1 inside bbox, 0 outside ─────────────────────────
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0

    # ── 4. Soften mask → smooth gradient boundary (no hard edge for VQ-VAE) ───
    mks = mask_ksize | 1
    mask_soft = cv2.GaussianBlur(mask, (mks, mks), 0)
    mask_3ch  = mask_soft[:, :, np.newaxis]          # (H, W, 1) for broadcasting

    # ── 5. Alpha-blend: sharp inside, blurred outside ─────────────────────────
    foveated = (mask_3ch * image + (1.0 - mask_3ch) * blurred).astype(np.uint8)
    return foveated


# ══════════════════════════════════════════════════════════════════════════════
# Saccade State Machine
# ══════════════════════════════════════════════════════════════════════════════

class SaccadeStateMachine:
    """
    Two-phase state machine that switches the fovea target based on gripper.

    GRASP phase  (gripper open):
        Sharp region = source object (what to pick up)
        DINO query   = source_noun  (e.g. "green block")

    PLACE phase  (gripper closed):
        Sharp region = destination  (where to put it)
        DINO query   = dest_noun    (e.g. "yellow block")

    The instant the gripper closes, the saccade fires: the blur centre jumps
    to the destination so the model can plan the placement trajectory.
    """

    GRASP = "grasp"
    PLACE = "place"

    def __init__(
        self,
        source_noun: str = "",
        dest_noun:   str = "",
        close_thresh: float = 0.5,    # gripper_norm ≥ this → PLACE phase
    ):
        self.source_noun  = source_noun
        self.dest_noun    = dest_noun
        self.close_thresh = close_thresh
        self._state       = self.GRASP

    def update(self, gripper_norm: float) -> bool:
        """
        Update phase from gripper value.
        Returns True if a phase transition just occurred (saccade fired).
        """
        prev = self._state
        self._state = (
            self.PLACE if gripper_norm >= self.close_thresh else self.GRASP
        )
        if prev != self._state:
            print(f"[Saccade] {prev} → {self._state}  "
                  f"(gripper={gripper_norm:.2f})")
            return True
        return False

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_target(self) -> str:
        """DINO text query for the current phase."""
        if self._state == self.PLACE and self.dest_noun:
            return self.dest_noun
        return self.source_noun

    def reset(self) -> None:
        self._state = self.GRASP

    def __repr__(self) -> str:
        return (f"SaccadeStateMachine(state={self._state}, "
                f"src='{self.source_noun}', dst='{self.dest_noun}')")


# ══════════════════════════════════════════════════════════════════════════════
# SaccadeFoveatedEmuVLAInference
# ══════════════════════════════════════════════════════════════════════════════

class SaccadeFoveatedEmuVLAInference(EmuVLAInference):
    """
    Foveated-blur preprocessing with saccade phase switching.

    Design rationale
    ----------------
    - Full image size / token count unchanged (zero-shot compatible).
    - No token mixing, no crop, no coordinate-system shift.
    - DINO bbox → soft-mask blur → single natural-looking image → VQ-VAE.
    - The model's own attention mechanism concentrates on the sharp bbox
      region naturally (high-frequency information = more attention weight).
    - Gripper state drives a saccade: blur centre jumps from source object
      to destination object the moment the gripper closes.

    Parameters
    ----------
    blur_ksize    : Gaussian kernel size for the background (default 61).
                    Larger = more aggressive background suppression.
    mask_ksize    : Kernel for mask edge softening (default 41).
                    Larger = wider gradient transition zone.
    bbox_margin   : Fractional expansion of DINO bbox (default 0.15 = 15 %).
    dino_cache_steps : Reuse DINO detection for N steps (speed).
    close_thresh  : Normalised gripper value that triggers PLACE phase.
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        blur_ksize: int = 61,
        mask_ksize: int = 41,
        bbox_margin: float = 0.15,
        close_thresh: float = 0.5,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override  = fast_path
        self._current_instruction = ""
        self._blur_ksize          = blur_ksize
        self._mask_ksize          = mask_ksize
        self._bbox_margin         = bbox_margin

        # DINO wrapper (bbox detection)
        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )

        # Saccade state machine (populated from instruction in step())
        self.saccade = SaccadeStateMachine(close_thresh=close_thresh)

        # BBox cache (reset on saccade transition)
        self._cached_bbox: Optional[Tuple[int,int,int,int]] = None
        self._cache_step: int = 0
        self._cache_steps: int = dino_cache_steps

        super().__init__(
            emu_hub=emu_hub,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
            policy_setup=policy_setup,
        )

    # ── bbox detection with N-step cache ──────────────────────────────────────

    def _get_bbox(self, image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Return cached DINO bbox or run fresh detection."""
        target = self.saccade.current_target
        if not target:
            return None

        # Cache hit
        if self._cached_bbox is not None and self._cache_step % self._cache_steps != 0:
            self._cache_step += 1
            return self._cached_bbox

        # Fresh detection
        try:
            bbox = self.dino.detect_bbox(image, target)
        except Exception as e:
            print(f"[Saccade] DINO error: {e}")
            bbox = None

        self._cached_bbox = bbox
        self._cache_step += 1
        return bbox

    # ── preprocess ────────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray):
        """
        Apply foveated blur and encode.

        If DINO finds the target:
            → sharp bbox, blurred periphery
        If DINO fails (no detection):
            → original image unchanged (safe fallback)
        """
        bbox = self._get_bbox(image)

        if bbox is not None:
            foveated = apply_foveated_blur(
                image, bbox,
                blur_ksize=self._blur_ksize,
                mask_ksize=self._mask_ksize,
                bbox_margin=self._bbox_margin,
            )
        else:
            # Fallback: pass original image; no information lost
            print(f"[Saccade] No bbox for '{self.saccade.current_target}', "
                  "using original image.")
            foveated = image

        agent_view = Image.fromarray(foveated).resize(self.image_size)
        dev = next(self.model.parameters()).device
        image_x = self.image_processor(agent_view, return_tensors="pt")[
            "pixel_values"
        ].to(dev)
        return self.image_tokenizer.encode(image_x), None

    # ── step: update instruction parse + gripper phase ────────────────────────

    def step(self, image: np.ndarray, goal: str):
        # Parse source / destination from instruction on change
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun   = dst
            print(f"[Saccade] Instruction parsed: src='{src}' dst='{dst}'")

        raw_actions, env_actions = super().step(image, goal)

        # Update saccade phase from gripper action
        if env_actions:
            g = float(np.asarray(env_actions[-1].get("gripper", [1.0])).flat[0])
            gripper_norm = (1.0 - g) / 2.0    # +1=open→0, -1=close→1
            transitioned = self.saccade.update(gripper_norm)
            if transitioned:
                # Saccade fired: flush bbox cache so DINO re-detects new target
                self._cached_bbox = None
                self._cache_step  = 0

        return raw_actions, env_actions

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""
        self._cached_bbox         = None
        self._cache_step          = 0
        self.saccade.reset()


# ══════════════════════════════════════════════════════════════════════════════
# Token-level Saccade helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_to_token_mask(
    bbox: Tuple[int, int, int, int],
    H_t: int, W_t: int,
    H_img: int, W_img: int,
    margin: int = 2,
) -> torch.BoolTensor:
    """
    Convert pixel bbox (x1,y1,x2,y2) → boolean fovea mask on (H_t, W_t) token grid.

    `margin` expands the bbox by that many tokens in each direction so the
    object edges are never accidentally included in the peripheral zone.
    """
    x1, y1, x2, y2 = bbox
    tx1 = max(0,   int(x1 * W_t / W_img) - margin)
    ty1 = max(0,   int(y1 * H_t / H_img) - margin)
    tx2 = min(W_t, int(x2 * W_t / W_img) + margin)
    ty2 = min(H_t, int(y2 * H_t / H_img) + margin)
    mask = torch.zeros(H_t, W_t, dtype=torch.bool)
    mask[ty1:ty2, tx1:tx2] = True
    return mask


def _progressive_soft_mask(
    tokens: torch.Tensor,         # (1, H_t, W_t) int64
    fovea_mask: torch.BoolTensor, # (H_t, W_t)  True = keep sharp
    near_pool: int = 3,
    far_pool: int = 7,
    near_expand: int = 2,
) -> torch.Tensor:
    """
    Progressive soft-masking of peripheral tokens in discrete VQ token space.

    Three zones
    -----------
    Fovea          — original tokens, 100% sharp.
    Near-periphery — mode of (near_pool × near_pool) neighbourhood.
                     Object silhouettes, coarse shapes still readable.
    Far-periphery  — mode of (far_pool × far_pool) neighbourhood.
                     Only dominant colour / region survives — basket "blob" visible.

    All replacements are valid VQ-codebook entries → zero distribution shift.
    The basket (or source cube) is never erased; it becomes a blurry presence,
    enough for the model to know "yellow thing is over there."
    """
    H_t, W_t = tokens.shape[-2:]

    # ── dilate fovea mask to define the near-periphery transition zone ─────
    dilation_k = near_expand * 2 + 1
    fov_f = fovea_mask.float().unsqueeze(0).unsqueeze(0)           # (1,1,H_t,W_t)
    near_zone = (
        F.max_pool2d(fov_f, kernel_size=dilation_k, stride=1, padding=near_expand)
        .squeeze() > 0
    )                                                               # fovea + margin
    far_zone = ~near_zone

    def _pool_mode(pool_k: int) -> torch.Tensor:
        """Local mode of pool_k×pool_k neighbourhood for every token position."""
        pad   = pool_k // 2
        t_pad = F.pad(
            tokens[0].float().unsqueeze(0).unsqueeze(0),
            [pad] * 4, mode="reflect",
        )
        patches = t_pad.unfold(2, pool_k, 1).unfold(3, pool_k, 1)
        flat    = patches.reshape(H_t, W_t, pool_k * pool_k).long()
        return flat.mode(dim=-1).values                            # (H_t, W_t)

    near_smooth = _pool_mode(near_pool)
    far_smooth  = _pool_mode(far_pool)

    tokens_out = tokens.clone()
    near_periphery = near_zone & ~fovea_mask
    tokens_out[0][near_periphery] = near_smooth[near_periphery]
    tokens_out[0][far_zone]       = far_smooth[far_zone]

    return tokens_out


# ══════════════════════════════════════════════════════════════════════════════
# TokenSaccadeEmuVLAInference
# ══════════════════════════════════════════════════════════════════════════════

class TokenSaccadeEmuVLAInference(EmuVLAInference):
    """
    Token-level Saccade Attention — Zero-shot.

    Strongest zero-shot pipeline: safest encoding (clean image → VQ-VAE)
    + bbox-precision fovea + progressive soft peripheral pooling + saccade.

    Phase 1 — GRASP (gripper open)
    --------------------------------
    DINO target : source object  (e.g. "green block")
    Fovea zone  : source bbox tokens  → 100% original (sharp, precise)
    Near-periph : 3×3 local-mode pool → coarse shapes preserved
    Far-periph  : 7×7 local-mode pool → basket visible as colour blob
    Effect      : model concentrates on cube for precise grasp;
                  basket location is still encoded (won't be forgotten)

    Phase 2 — PLACE  (gripper closes → saccade fires instantly)
    -------------------------------------------------------------
    DINO target : destination object  (e.g. "yellow block" / "basket")
    Fovea zone  : destination bbox tokens → 100% original
    Near/far    : source object area softly pooled (still present, not dominant)
    Effect      : model attention jumps to destination;
                  held cube visible as a blob so model knows it carries something

    Key properties
    --------------
    ✓ VQ-VAE always sees clean unmodified pixels         (zero image-level OOD)
    ✓ All peripheral replacements are VQ-codebook tokens (zero token-level OOD)
    ✓ Destination NEVER disappears during Phase 1        (soft pool ≠ erase)
    ✓ Saccade is instant: bbox cache flushed on gripper close
    ✓ No resize / crop / token-count change              (architecture-safe)
    """

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        near_pool: int = 3,       # neighbourhood size for near-periphery pooling
        far_pool: int = 7,        # neighbourhood size for far-periphery pooling
        near_expand: int = 2,     # token-margin around fovea bbox
        bbox_margin: int = 2,     # token-margin when converting bbox → mask
        close_thresh: float = 0.5,
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override  = fast_path
        self._current_instruction = ""
        self._near_pool           = near_pool
        self._far_pool            = far_pool
        self._near_expand         = near_expand
        self._bbox_margin         = bbox_margin

        self.dino = GroundingDINOWrapper(
            model_name=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            cache_steps=dino_cache_steps,
            debug_dir=dino_debug_dir,
        )
        self.saccade = SaccadeStateMachine(close_thresh=close_thresh)

        self._bbox_cache: Optional[Tuple[int, int, int, int]] = None
        self._cache_step: int  = 0
        self._cache_steps: int = dino_cache_steps

        super().__init__(
            emu_hub=emu_hub, vq_hub=vq_hub, vision_hub=vision_hub,
            device=device, policy_setup=policy_setup,
        )

    # ── DINO bbox with N-step cache ────────────────────────────────────────

    def _get_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        target = self.saccade.current_target
        if not target:
            return None
        if self._bbox_cache is not None and self._cache_step % self._cache_steps != 0:
            self._cache_step += 1
            return self._bbox_cache
        try:
            bbox = self.dino.detect_bbox(image, target)
        except Exception as e:
            print(f"[TokenSaccade] DINO error: {e}")
            bbox = None
        self._bbox_cache = bbox
        self._cache_step += 1
        return bbox

    # ── preprocess ────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray):
        H, W  = image.shape[:2]
        dev   = next(self.model.parameters()).device

        # ── Step 1: encode clean image → tokens (no distribution shift) ───
        agent_view = Image.fromarray(image).resize(self.image_size)
        img_x = self.image_processor(agent_view, return_tensors="pt")[
            "pixel_values"
        ].to(dev)
        tokens = self.image_tokenizer.encode(img_x)          # (1, H_t, W_t)
        H_t, W_t = tokens.shape[-2], tokens.shape[-1]

        # ── Step 2: DINO bbox → token-grid fovea mask ─────────────────────
        bbox = self._get_bbox(image)
        if bbox is None:
            print(f"[TokenSaccade] No bbox for '{self.saccade.current_target}'"
                  " — all tokens kept.")
            return tokens, None

        fovea_mask = _bbox_to_token_mask(
            bbox, H_t, W_t, H, W, margin=self._bbox_margin
        )

        # ── Step 3: progressive soft-mask periphery ────────────────────────
        tokens_out = _progressive_soft_mask(
            tokens, fovea_mask,
            near_pool=self._near_pool,
            far_pool=self._far_pool,
            near_expand=self._near_expand,
        )

        n_fovea = int(fovea_mask.sum())
        print(f"[TokenSaccade] phase={self.saccade.state} "
              f"target='{self.saccade.current_target}' "
              f"fovea={n_fovea}/{H_t * W_t} tokens  bbox={bbox}")
        return tokens_out, None

    # ── step: sync instruction parse + gripper phase ──────────────────────

    def step(self, image: np.ndarray, goal: str):
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun   = dst
            print(f"[TokenSaccade] Instruction → src='{src}'  dst='{dst}'")

        raw_actions, env_actions = super().step(image, goal)

        if env_actions:
            g = float(np.asarray(env_actions[-1].get("gripper", [1.0])).flat[0])
            gripper_norm = (1.0 - g) / 2.0        # +1=open→0, −1=close→1
            transitioned = self.saccade.update(gripper_norm)
            if transitioned:
                # Saccade fired: flush cache → DINO re-detects new target next step
                self._bbox_cache = None
                self._cache_step = 0

        return raw_actions, env_actions

    def reset(self) -> None:
        super().reset()
        self.dino.reset()
        self._current_instruction = ""
        self._bbox_cache          = None
        self._cache_step          = 0
        self.saccade.reset()
