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
