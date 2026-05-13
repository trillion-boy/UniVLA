#!/usr/bin/env python3
"""
LatentSaccade — Standalone Single-File Implementation
======================================================
Embedding-level foveated attention for UniVLA (Emu3-based VLA).

핵심 아이디어:
  - VQ-VAE, token ID, LLM 가중치 일체 변경 없음 (inference-only)
  - embed_tokens 출력 직후 visual token embedding에 spatial weight 적용
    fovea(DINO bbox) = 1.0 / secondary(손에 든 물체) = 0.5 / background = 0.2
  - Saccade: gripper 상태에 따라 GRASP→PLACE phase 자동 전환

실행 방법 (Colab):
  !MUJOCO_GL=osmesa xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
      conda run -n univla --no-capture-output \
      python latent_saccade_standalone.py \
          --emu-hub      /content/pretrain/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K \
          --vq-hub       /content/pretrain/Emu3-VisionTokenizer \
          --fast-path    /content/UniVLA/pretrain/fast_bridge_t5_s50 \
          --task         widowx_stack_cube \
          --n-episodes   24 \
          --output-dir   /content/latent_saccade_results \
          --bg-weight    0.2 \
          --place-src-weight 0.5 \
          --save-video
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0. Path & Compatibility Patches
# ══════════════════════════════════════════════════════════════════════════════

import sys, os, types, argparse, json, time, re
from queue import Queue
from typing import Optional, Tuple

ROOT     = "/content/UniVLA"
EXP      = os.path.join(ROOT, "experiments", "foveated_tokenization")
EMU3     = os.path.join(ROOT, "reference", "Emu3")
SIMPLER  = "/content/SimplerEnv"
MANSKILL = os.path.join(SIMPLER, "ManiSkill2_real2sim")

for p in [ROOT, EXP, EMU3, SIMPLER, MANSKILL]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Patch 1: is_torch_fx_available (removed in newer transformers)
import transformers.utils.import_utils as _tui
if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: True

# Patch 2: ProcessorMixin type check (transformers 4.38.x vs 4.49+)
import transformers.processing_utils as _pu
if not getattr(_pu.ProcessorMixin, "_check_patched", False):
    _pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    _pu.ProcessorMixin._check_patched = True

# Patch 3: lightning stub (not installed in conda env)
if "lightning" not in sys.modules:
    for _n in ["lightning", "lightning.pytorch", "lightning.pytorch.trainer"]:
        sys.modules.setdefault(_n, types.ModuleType(_n))
    class _Trainer: pass
    sys.modules["lightning.pytorch.trainer"].Trainer = _Trainer

# Patch 4: Emu3Tokenizer.mergeable_ranks
from emu3.mllm import Emu3Tokenizer
if not hasattr(Emu3Tokenizer, "mergeable_ranks"):
    Emu3Tokenizer.mergeable_ranks = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Imports
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transforms3d.euler import euler2axangle
from transformers import (
    AutoImageProcessor, AutoModel, AutoProcessor,
    GenerationConfig, LogitsProcessor,
    AutoModelForZeroShotObjectDetection,
)
from transformers.feature_extraction_utils import BatchFeature
from emu3.mllm import Emu3MoE, Emu3Processor, Emu3Tokenizer

# ══════════════════════════════════════════════════════════════════════════════
# 2. GroundingDINOWrapper
#    Image + text query → DINO bbox (x1, y1, x2, y2) in pixel coordinates
# ══════════════════════════════════════════════════════════════════════════════

class GroundingDINOWrapper:
    """
    Grounding DINO wrapper for object detection.
    Returns bbox of the target object given an image and text query.
    Caches detection for dino_cache_steps steps to reduce compute.
    """

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
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        device: Optional[str] = None,
        cache_steps: int = 5,
        debug_dir: Optional[str] = None,
    ):
        self.model_name    = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_steps   = cache_steps
        self._debug_dir    = debug_dir
        self._model        = None
        self._processor    = None
        self._cache_step   = 0
        self._debug_count  = 0

    def _load_model(self):
        if self._model is not None:
            return
        print(f"[DINO] Loading {self.model_name} ...")
        from transformers import AutoProcessor as AP
        self._processor = AP.from_pretrained(self.model_name)
        self._model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.model_name)
            .to(self.device).eval()
        )
        print("[DINO] Loaded.")

    @staticmethod
    def extract_target_noun(instruction: str) -> str:
        instr = instruction.lower().strip()
        for pat in GroundingDINOWrapper._VERB_PATTERNS:
            m = re.search(pat, instr)
            if m:
                return m.group(1).strip()
        return instr

    @staticmethod
    def extract_source_dest_nouns(instruction: str) -> Tuple[str, Optional[str]]:
        instr = instruction.lower().strip()
        pat = r"(?:stack|put|place|move|transfer)\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto|to)\s+(?:the\s+)?(.+?)(?:\s*$|\s+and\s)"
        m = re.search(pat, instr)
        if m:
            return m.group(1).strip().rstrip(".,"), m.group(2).strip().rstrip(".,")
        return GroundingDINOWrapper.extract_target_noun(instruction), None

    def detect_bbox(self, image_rgb: np.ndarray, text_query: str) -> Optional[Tuple[int,int,int,int]]:
        """Returns (x1, y1, x2, y2) of highest-confidence detection, or None."""
        self._load_model()
        pil_img = Image.fromarray(image_rgb)
        query = text_query if text_query.endswith(".") else text_query + "."
        inputs = self._processor(images=pil_img, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]])
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_threshold, target_sizes=target_sizes,
            )[0]
        if len(results["boxes"]) == 0:
            return None
        best_idx = results["scores"].argmax().item()
        score = results["scores"][best_idx].item()
        box = results["boxes"][best_idx].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        print(f"[DINO] BBox '{text_query.rstrip('.')}' score={score:.3f} → [{x1},{y1},{x2},{y2}]")
        return x1, y1, x2, y2

    def reset(self):
        self._cache_step = 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. ActionIDConstraintLogitsProcessor
#    Constrains generation to valid action token IDs only
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
# 4. EmuVLAInference  (Baseline — no modification)
#    Copied from RoboVLMs/eval/simpler/model_wrapper.py, lightning deps removed
# ══════════════════════════════════════════════════════════════════════════════

class EmuVLAInference:
    def __init__(self, emu_hub, vq_hub, vision_hub, device,
                 policy_setup="widowx_bridge", fast_path=None):
        self.emu_hub      = emu_hub
        self.vq_hub       = vq_hub
        self.vision_hub   = vision_hub
        self.device       = device
        self.policy_setup = policy_setup
        if not hasattr(self, "_fast_path_override"):
            self._fast_path_override = fast_path

        if self.policy_setup == "google_robot":
            self.close_gripper_act = -1
            self.image_size = (160, 128)
        else:  # widowx_bridge
            self.close_gripper_act = 1
            self.image_size = (256, 256)

        self.sticky_gripper_num_repeat = 2
        self.sticky_action_is_on       = False
        self.gripper_action_repeat     = 0
        self.sticky_gripper_action     = 0.0
        self.previous_gripper_action   = None
        self.close_gripper_num         = 0
        self.late_close_gripper        = 1
        self.window_size               = 2
        self.predict_action_frames     = 5
        self.context_frames            = 1
        self.predict_frames            = 1
        self.action_dim                = 7
        self.use_gripper               = False
        self.use_fast                  = True
        self.use_one_step              = False
        self.eoa_token_id              = 151845
        self.use_cot                   = False
        self.video_mode                = True

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
                do_sample=True, top_k=2048, temperature=0.8,
            )

    def init_config(self, device):
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.emu_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right", use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        self.processor = Emu3Processor(
            image_processor=self.image_processor,
            vision_tokenizer=self.image_tokenizer,
            tokenizer=self.tokenizer,
        )

        _fast_base = getattr(self, "_fast_path_override", None) or "/share/project/yuqi.wang/UniVLA/pretrain"
        _direct_marker = ("processor_config.json", "tokenizer_config.json", "config.json")
        if any(os.path.isfile(os.path.join(_fast_base, f)) for f in _direct_marker):
            fast_path = _fast_base
        elif self.policy_setup == "widowx_bridge":
            fast_path = os.path.join(_fast_base, "fast_bridge_t5_s50")
        elif self.policy_setup == "google_robot":
            fast_path = os.path.join(_fast_base, "fast_google_a5_s50")
        else:
            fast_path = os.path.join(_fast_base, "fast")
        self.action_tokenizer = (
            AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)
            if os.path.isdir(fast_path) else None
        )

        self.rgb_list = []; self.hand_rgb_list = []; self.action_hist_list = []
        self.rollout_step_counter = 0
        self.vision_queue         = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue         = Queue(maxsize=self.window_size - 1)

    def add_image(self, image):
        if self.vision_queue.full(): self.vision_queue.get()
        self.vision_queue.put(image)

    def get_history(self): return list(self.vision_queue.queue)

    def add_action(self, action):
        if self.action_queue.full(): self.action_queue.get()
        self.action_queue.put(action)

    def get_action_history(self): return list(self.action_queue.queue)

    def reset(self):
        self.rgb_list = []; self.hand_rgb_list = []
        self.rollout_step_counter = 0; self.action_hist_list = []
        self.sticky_action_is_on = False; self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0; self.close_gripper_num = 0
        self.previous_gripper_action = None
        for q in [self.vision_queue, self.vision_gripper_queue, self.action_queue]:
            while not q.empty(): q.get()

    def preprocess(self, image: np.ndarray):
        agent_view = Image.fromarray(image).resize(self.image_size)
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        return self.image_tokenizer.encode(image_x), None

    def step(self, image: np.ndarray, goal: str):
        image_code, gripper_code = self.preprocess(image)
        video_code   = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_tokens = BatchFeature(
            data={**self.processor.tokenizer([self.tokenizer.bos_token + goal])},
            tensor_type="pt"
        )
        pos_inputs = self.processor.video_process(
            text=goal, video_tokens=video_code, gripper_tokens=gripper_code,
            context_frames=self.context_frames, frames=self.predict_frames,
            return_tensors="pt", mode="VLA_Video", padding="longest",
        )

        if self.video_mode:
            self.add_image(pos_inputs)
            history = self.get_history(); action_history = self.get_action_history()
            all_ids, all_types, all_masks = [text_tokens["input_ids"]], [text_tokens["token_type_ids"]], [text_tokens["attention_mask"]]
            for i, hist in enumerate(history):
                if i < len(action_history):
                    act = action_history[i]
                    all_ids.extend([hist["input_ids"], act])
                    all_types.extend([hist["token_type_ids"], torch.zeros_like(act)])
                    all_masks.extend([hist["attention_mask"], torch.ones_like(act)])
                else:
                    all_ids.append(hist["input_ids"]); all_types.append(hist["token_type_ids"]); all_masks.append(hist["attention_mask"])
            final_inputs = pos_inputs.copy()
            final_inputs["input_ids"]      = torch.cat(all_ids,   dim=1)
            final_inputs["token_type_ids"] = torch.cat(all_types, dim=1)
            final_inputs["attention_mask"] = torch.cat(all_masks, dim=1)
        else:
            final_inputs = pos_inputs

        last_token_id = self.tokenizer.pad_token_id - 1
        allowed = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
        with torch.no_grad():
            outputs = self.model.generate(
                final_inputs.input_ids.to(self.device), self.GENERATION_CONFIG,
                max_new_tokens=100, logits_processor=[ActionIDConstraintLogitsProcessor(allowed)],
                attention_mask=final_inputs.attention_mask.to(self.device),
            )
        orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
        processed    = torch.tensor(last_token_id, dtype=orig_outputs.dtype, device=orig_outputs.device) - orig_outputs[:, :-1]
        action = self.action_tokenizer.decode(processed, time_horizon=self.predict_action_frames, action_dim=self.action_dim)[0]
        if self.video_mode: self.add_action(orig_outputs.detach().cpu())
        action = self.unormalize_action(action)
        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        return [r[0] for r in res], [r[1] for r in res]

    def unormalize_action(self, action):
        if self.policy_setup == "google_robot":
            high = np.array([0.17753939667156038,0.14669284061245857,0.2179850059450077,0.5882710857765758,0.35334834816471683,0.4470693223284772,0.99980000009996])
            low  = np.array([-0.22624227067544056,-0.15126218201085617,-0.23251856873127252,-0.3538952427136002,-0.4202906595250484,-0.43766197340888247,-1e-10])
        else:
            high = np.array([0.02819482065335177,0.04079562563528227,0.04015785568112329,0.08070396399877966,0.07745134926258301,0.2016542930635028,0.99980000009996])
            low  = np.array([-0.02887803485105428,-0.04178320091122349,-0.026113155505000457,-0.08117201867235568,-0.09309056401752747,-0.20778717060421048,-1e-10])
        return 0.5 * (action + 1) * (high - low) + low

    def transform_action(self, raw_actions):
        raw_action = {"world_vector": np.array(raw_actions[0,:3]), "rotation_delta": np.array(raw_actions[0,3:6]), "open_gripper": np.array(raw_actions[0,6:7])}
        action = {"world_vector": raw_action["world_vector"]}
        roll, pitch, yaw = raw_action["rotation_delta"]
        ax, angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = ax * angle
        if self.policy_setup == "google_robot":
            cur = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            rel = np.array([0]) if self.previous_gripper_action is None else self.previous_gripper_action - cur
            self.previous_gripper_action = cur
            if np.abs(rel) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True; self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1; rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False; self.gripper_action_repeat = 0; self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        else:  # widowx_bridge
            rel = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            if rel[0] > 0: self.close_gripper_num += 1
            else: self.close_gripper_num = 0
            rel[0] = 1 if self.close_gripper_num >= self.late_close_gripper else -1
            action["gripper"] = rel
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action


# ══════════════════════════════════════════════════════════════════════════════
# 5. SaccadeStateMachine
#    GRASP phase → gripper closes → PLACE phase → gripper opens → GRASP
# ══════════════════════════════════════════════════════════════════════════════

class SaccadeStateMachine:
    GRASP = "grasp"
    PLACE = "place"

    def __init__(self, source_noun="", dest_noun="",
                 close_thresh=0.5, min_grasp_steps=15, consecutive_close_required=3):
        self.source_noun = source_noun
        self.dest_noun   = dest_noun
        self.close_thresh = close_thresh
        self.min_grasp_steps = min_grasp_steps
        self.consecutive_close_required = consecutive_close_required
        self._state       = self.GRASP
        self._grasp_steps = 0
        self._close_count = 0

    def update(self, gripper_norm: float) -> bool:
        transitioned = False
        if self._state == self.GRASP:
            self._grasp_steps += 1
            self._close_count = self._close_count + 1 if gripper_norm >= self.close_thresh else 0
            if self._grasp_steps >= self.min_grasp_steps and self._close_count >= self.consecutive_close_required:
                self._state = self.PLACE; self._grasp_steps = 0; self._close_count = 0
                transitioned = True
                print(f"[Saccade] grasp → place (gripper={gripper_norm:.2f})")
        else:
            if gripper_norm < self.close_thresh:
                self._state = self.GRASP; self._grasp_steps = 0; self._close_count = 0
                transitioned = True
                print(f"[Saccade] place → grasp (gripper={gripper_norm:.2f})")
        return transitioned

    @property
    def state(self): return self._state

    @property
    def current_target(self):
        return self.dest_noun if self._state == self.PLACE and self.dest_noun else self.source_noun

    def reset(self):
        self._state = self.GRASP; self._grasp_steps = 0; self._close_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Helper: pixel bbox → token grid mask
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_to_token_mask(bbox, H_t, W_t, H_img, W_img, margin=2):
    """Convert pixel bbox (x1,y1,x2,y2) → boolean mask on (H_t, W_t) token grid."""
    x1, y1, x2, y2 = bbox
    tx1 = max(0,   int(x1 * W_t / W_img) - margin)
    ty1 = max(0,   int(y1 * H_t / H_img) - margin)
    tx2 = min(W_t, int(x2 * W_t / W_img) + margin)
    ty2 = min(H_t, int(y2 * H_t / H_img) + margin)
    mask = torch.zeros(H_t, W_t, dtype=torch.bool)
    mask[ty1:ty2, tx1:tx2] = True
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 7. LatentSaccadeEmuVLAInference  ← 핵심 클래스
#
# Pipeline:
#   Image → VQ-VAE → 1024 token IDs → embed_tokens → [개입] → LLM → Action
#                                                         ↑
#                                    DINO bbox → weight map (32×32)
#                                    fovea=1.0 / secondary=0.5 / bg=0.2
#
# Key properties:
#   - VQ-VAE, token IDs, LLM weights unchanged (zero OOD)
#   - Only embedding magnitudes are scaled
#   - embed_tokens forward hook fires only on first full-sequence call
#   - Saccade switches fovea target: GRASP→source, PLACE→destination
# ══════════════════════════════════════════════════════════════════════════════

class LatentSaccadeEmuVLAInference(EmuVLAInference):

    def __init__(
        self,
        emu_hub: str,
        vq_hub: str,
        vision_hub: str,
        device: str,
        policy_setup: str = "widowx_bridge",
        fast_path: Optional[str] = None,
        # DINO settings
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        bbox_margin: int = 2,
        # Weight map settings — key hyperparameters to tune
        bg_weight: float = 0.2,           # background embedding scale (0~1)
        place_src_weight: float = 0.5,    # secondary object scale in PLACE phase
        # Saccade settings
        close_thresh: float = 0.5,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        # Ablation
        enable_latent_mask: bool = True,  # set False to disable (baseline mode)
        dino_debug_dir: Optional[str] = None,
    ):
        self._fast_path_override  = fast_path
        self._current_instruction = ""
        self._bg_weight           = bg_weight
        self._place_src_weight    = place_src_weight
        self._bbox_margin         = bbox_margin
        self._enable_latent_mask  = enable_latent_mask

        self.dino = GroundingDINOWrapper(
            model_name=dino_model, box_threshold=box_threshold,
            text_threshold=text_threshold, device=device,
            cache_steps=dino_cache_steps, debug_dir=dino_debug_dir,
        )
        self.saccade = SaccadeStateMachine(
            close_thresh=close_thresh,
            min_grasp_steps=min_grasp_steps,
            consecutive_close_required=consecutive_close_required,
        )
        self._fovea_bbox_cache:     Optional[Tuple] = None
        self._secondary_bbox_cache: Optional[Tuple] = None
        self._cache_step:  int = 0
        self._cache_steps: int = dino_cache_steps

        super().__init__(emu_hub=emu_hub, vq_hub=vq_hub, vision_hub=vision_hub,
                         device=device, policy_setup=policy_setup)
        self._setup_vis_range()

    def _setup_vis_range(self):
        """Derive VQ visual token ID range from tokenizer vocabulary."""
        tmpl = getattr(self.processor, "visual_template", None)
        fmt  = tmpl[0] if tmpl else "<|visual token {token_id:0>6d}|>"
        self.vis_start = self.tokenizer.convert_tokens_to_ids(fmt.format(token_id=0))
        self.vis_end   = self.tokenizer.convert_tokens_to_ids(fmt.format(token_id=32767))

    def _get_bboxes(self, image: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Returns (fovea_bbox, secondary_bbox) with N-step caching."""
        use_cache = (self._fovea_bbox_cache is not None and self._cache_step % self._cache_steps != 0)
        if use_cache:
            self._cache_step += 1
            return self._fovea_bbox_cache, self._secondary_bbox_cache

        target = self.saccade.current_target
        fovea_bbox = self.dino.detect_bbox(image, target) if target else None

        secondary_bbox = None
        if self.saccade.state == SaccadeStateMachine.PLACE and self.saccade.source_noun:
            secondary_bbox = self.dino.detect_bbox(image, self.saccade.source_noun)

        self._fovea_bbox_cache    = fovea_bbox
        self._secondary_bbox_cache = secondary_bbox
        self._cache_step += 1
        return fovea_bbox, secondary_bbox

    def _build_weight_map(self, image, fovea_bbox, secondary_bbox, H_t=32, W_t=32):
        """
        Build (H_t * W_t,) weight vector.
        fovea=1.0 > secondary=place_src_weight > background=bg_weight
        Returns None when no bbox available (masking disabled gracefully).
        """
        if fovea_bbox is None and secondary_bbox is None:
            return None
        H, W = image.shape[:2]
        weight = torch.full((H_t, W_t), self._bg_weight)
        if secondary_bbox is not None:
            weight[_bbox_to_token_mask(secondary_bbox, H_t, W_t, H, W, self._bbox_margin)] = self._place_src_weight
        if fovea_bbox is not None:
            weight[_bbox_to_token_mask(fovea_bbox, H_t, W_t, H, W, self._bbox_margin)] = 1.0
        return weight.reshape(-1)  # (1024,)

    def _make_embed_hook(self, weight_1d: Optional[torch.Tensor], frame_start: int):
        """
        One-shot forward hook on embed_tokens.
        Applies spatial weight map to current-frame visual tokens only.
        Skips subsequent single-token generation calls automatically.
        """
        fired = [False]
        def hook(module, inp, output):
            if fired[0] or inp[0].shape[1] <= 1:
                return output
            fired[0] = True
            if not self._enable_latent_mask or weight_1d is None:
                return output
            frame_ids = inp[0][0, frame_start:]
            is_visual = (frame_ids >= self.vis_start) & (frame_ids <= self.vis_end)
            vis_idx   = is_visual.nonzero(as_tuple=True)[0]
            if vis_idx.numel() == 0:
                return output
            w = weight_1d[:vis_idx.numel()].to(dtype=output.dtype, device=output.device)
            new_out = output.clone()
            new_out[0, frame_start + vis_idx] = output[0, frame_start + vis_idx] * w.unsqueeze(-1)
            return new_out
        return hook

    def step(self, image: np.ndarray, goal: str):
        # 1. Parse instruction → saccade source/destination nouns
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun   = dst
            print(f"[LatentSaccade] Instruction → src='{src}'  dst='{dst}'")

        # 2. DINO detection → weight map
        fovea_bbox, secondary_bbox = self._get_bboxes(image)
        weight_1d = self._build_weight_map(image, fovea_bbox, secondary_bbox)

        # 3. Clean VQ-VAE encode (no pixel modification)
        image_code, gripper_code = self.preprocess(image)

        # 4. Build full input sequence
        video_code   = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None
        text_tokens  = BatchFeature(
            data={**self.processor.tokenizer([self.tokenizer.bos_token + goal])}, tensor_type="pt"
        )
        pos_inputs = self.processor.video_process(
            text=goal, video_tokens=video_code, gripper_tokens=gripper_code,
            context_frames=self.context_frames, frames=self.predict_frames,
            return_tensors="pt", mode="VLA_Video", padding="longest",
        )
        if self.video_mode:
            self.add_image(pos_inputs)
            history = self.get_history(); action_history = self.get_action_history()
            all_ids, all_types, all_masks = [text_tokens["input_ids"]], [text_tokens["token_type_ids"]], [text_tokens["attention_mask"]]
            for i, hist in enumerate(history):
                if i < len(action_history):
                    act = action_history[i]
                    all_ids.extend([hist["input_ids"], act])
                    all_types.extend([hist["token_type_ids"], torch.zeros_like(act)])
                    all_masks.extend([hist["attention_mask"], torch.ones_like(act)])
                else:
                    all_ids.append(hist["input_ids"]); all_types.append(hist["token_type_ids"]); all_masks.append(hist["attention_mask"])
            final_inputs = pos_inputs.copy()
            final_inputs["input_ids"]      = torch.cat(all_ids,   dim=1)
            final_inputs["token_type_ids"] = torch.cat(all_types, dim=1)
            final_inputs["attention_mask"] = torch.cat(all_masks, dim=1)
        else:
            final_inputs = pos_inputs

        # 5. Locate current frame position in full sequence
        current_frame_len = pos_inputs["input_ids"].shape[1]
        frame_start       = final_inputs["input_ids"].shape[1] - current_frame_len

        n_fovea = int((weight_1d >= 1.0).sum()) if weight_1d is not None else 0
        n_bg    = int((weight_1d < self._place_src_weight).sum()) if weight_1d is not None else 0
        print(f"[LatentSaccade] phase={self.saccade.state}  target='{self.saccade.current_target}'  "
              f"fovea_tokens≈{n_fovea}  bg_tokens≈{n_bg}  fovea_bbox={fovea_bbox}")

        # 6. Register embed_tokens hook → generate → remove hook
        embed_module = self.model.get_input_embeddings()
        handle = embed_module.register_forward_hook(self._make_embed_hook(weight_1d, frame_start))

        last_token_id = self.tokenizer.pad_token_id - 1
        allowed = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device), self.GENERATION_CONFIG,
                    max_new_tokens=100, logits_processor=[ActionIDConstraintLogitsProcessor(allowed)],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )
        finally:
            handle.remove()  # always remove hook even if generation fails

        # 7. Decode actions
        orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
        processed    = torch.tensor(last_token_id, dtype=orig_outputs.dtype, device=orig_outputs.device) - orig_outputs[:, :-1]
        action = self.action_tokenizer.decode(processed, time_horizon=self.predict_action_frames, action_dim=self.action_dim)[0]
        if self.video_mode: self.add_action(orig_outputs.detach().cpu())
        action = self.unormalize_action(action)
        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        raw_actions, env_actions = [r[0] for r in res], [r[1] for r in res]

        # 8. Update saccade phase from gripper output
        if env_actions:
            g = float(np.asarray(env_actions[-1].get("gripper", [1.0])).flat[0])
            transitioned = self.saccade.update((1.0 - g) / 2.0)
            if transitioned:
                self._fovea_bbox_cache = None; self._secondary_bbox_cache = None; self._cache_step = 0

        return raw_actions, env_actions

    def reset(self):
        super().reset()
        self.dino.reset()
        self._current_instruction    = ""
        self._fovea_bbox_cache       = None
        self._secondary_bbox_cache   = None
        self._cache_step             = 0
        self.saccade.reset()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation Script
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LatentSaccade SimplerEnv Evaluation")
    p.add_argument("--emu-hub",    required=True,  help="Path to UniVLA checkpoint")
    p.add_argument("--vq-hub",     required=True,  help="Path to Emu3-VisionTokenizer")
    p.add_argument("--fast-path",  required=True,  help="Path to fast action tokenizer")
    p.add_argument("--task",       default="widowx_put_eggplant_in_basket",
                   choices=["widowx_put_eggplant_in_basket", "widowx_carrot_on_plate", "widowx_stack_cube"])
    p.add_argument("--n-episodes",        type=int,   default=24)
    p.add_argument("--output-dir",        default="/content/latent_saccade_results")
    # Weight map hyperparameters
    p.add_argument("--bg-weight",         type=float, default=0.2,  help="Background embedding scale")
    p.add_argument("--place-src-weight",  type=float, default=0.5,  help="Secondary object scale in PLACE phase")
    # Saccade hyperparameters
    p.add_argument("--min-grasp-steps",   type=int,   default=15,   help="Min steps before GRASP→PLACE transition")
    p.add_argument("--consec-close",      type=int,   default=3,    help="Consecutive gripper-close steps to trigger saccade")
    # DINO hyperparameters
    p.add_argument("--dino-cache-steps",  type=int,   default=5,    help="Reuse DINO detection for N steps")
    p.add_argument("--box-threshold",     type=float, default=0.15)
    p.add_argument("--text-threshold",    type=float, default=0.15)
    p.add_argument("--bbox-margin",       type=int,   default=2,    help="Token grid expansion around DINO bbox")
    # Misc
    p.add_argument("--save-video",        action="store_true")
    p.add_argument("--dino-debug-dir",    default=None)
    p.add_argument("--disable-latent-mask", action="store_true", help="Ablation: disable weight masking (baseline mode)")
    return p.parse_args()


TASK_CONFIGS = {
    "widowx_put_eggplant_in_basket": {
        "env_name":         "PutEggplantInBasketScene-v0",
        "robot":            "widowx_sink_camera_setup",
        "scene_name":       "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name":  "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 120,
    },
    "widowx_carrot_on_plate": {
        "env_name":         "PutCarrotOnPlateInScene-v0",
        "robot":            "widowx",
        "scene_name":       "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name":  "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
    "widowx_stack_cube": {
        "env_name":         "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot":            "widowx",
        "scene_name":       "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name":  "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
}


def build_env(cfg, ep_id):
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    robot = cfg["robot"]
    kw = dict(
        obs_mode="rgbd", robot=robot,
        sim_freq=cfg["sim_freq"], control_mode=get_robot_control_mode(robot, "emu_vla"),
        control_freq=cfg["control_freq"], max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"], camera_cfgs={"add_segmentation": True},
    )
    for base in [SIMPLER, os.path.join(SIMPLER, "ManiSkill2_real2sim")]:
        cand = os.path.join(base, cfg["rgb_overlay_path"])
        if os.path.exists(cand):
            kw["rgb_overlay_path"]    = cand
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
            break
    env = build_maniskill2_env(cfg["env_name"], **kw)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    return env, obs


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def main():
    args     = parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] LatentSaccadeEmuVLAInference  task={args.task}", flush=True)
    print(f"       bg_weight={args.bg_weight}  place_src_weight={args.place_src_weight}", flush=True)
    model = LatentSaccadeEmuVLAInference(
        emu_hub=args.emu_hub, vq_hub=args.vq_hub, vision_hub=args.vq_hub,
        device="cuda", policy_setup="widowx_bridge", fast_path=args.fast_path,
        dino_model="IDEA-Research/grounding-dino-tiny",
        dino_cache_steps=args.dino_cache_steps,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        bbox_margin=args.bbox_margin,
        bg_weight=args.bg_weight, place_src_weight=args.place_src_weight,
        min_grasp_steps=args.min_grasp_steps,
        consecutive_close_required=args.consec_close,
        enable_latent_mask=not args.disable_latent_mask,
        dino_debug_dir=args.dino_debug_dir,
    )
    print(f"[OK] Model loaded  vis_start={model.vis_start}  vis_end={model.vis_end}", flush=True)

    base_ids = list(range(*task_cfg["obj_episode_range"]))
    ep_ids   = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results  = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────", flush=True)
        env, obs    = build_env(task_cfg, ep_id)
        instruction = env.get_language_instruction()
        image       = get_image(env, obs, cam_name)
        print(f"   instruction: {instruction}", flush=True)

        model.reset()
        frames = [image.copy()] if args.save_video else []
        done = truncated = False
        step = 0; t0 = time.time()

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            raw_actions, env_actions = model.step(image, instruction)
            for raw_a, env_a in zip(raw_actions, env_actions):
                obs, _, done, truncated, _ = env.step(np.concatenate([
                    env_a["world_vector"], env_a["rot_axangle"], env_a["gripper"],
                ]))
                image = get_image(env, obs, cam_name)
                if args.save_video and step % 4 == 0:
                    frames.append(image.copy())
                new_instr = env.get_language_instruction()
                if new_instr != instruction:
                    instruction = new_instr; model.reset()
                step += 1
                if done or truncated: break

        elapsed = time.time() - t0
        status  = "SUCCESS" if done else "FAIL"
        print(f"   → {status}  ({step} steps, {elapsed:.1f}s)", flush=True)
        env.close()

        if args.save_video and frames:
            from PIL import Image as _PIL
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            pils  = [_PIL.fromarray(f) for f in frames]
            pils[0].save(vpath, save_all=True, append_images=pils[1:], loop=0, duration=100)
            print(f"   GIF: {vpath}", flush=True)

        results.append({"ep": ep_count, "ep_id": ep_id, "success": bool(done), "steps": step, "elapsed": elapsed})

    n_ok = sum(r["success"] for r in results)
    sr   = n_ok / len(results)
    print(f"\n{'='*50}", flush=True)
    print(f"  task: {args.task}", flush=True)
    print(f"  Success rate: {n_ok}/{len(results)} = {sr:.1%}", flush=True)
    print(f"  Avg steps: {np.mean([r['steps'] for r in results]):.0f}", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        print(f"  {'✓' if r['success'] else '✗'} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps", flush=True)

    summary = {
        "model": "LatentSaccade",
        "task": args.task,
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "config": {
            "bg_weight": args.bg_weight,
            "place_src_weight": args.place_src_weight,
            "min_grasp_steps": args.min_grasp_steps,
            "consec_close": args.consec_close,
            "dino_cache_steps": args.dino_cache_steps,
            "enable_latent_mask": not args.disable_latent_mask,
        },
        "episodes": results,
    }
    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {save_path}", flush=True)


if __name__ == "__main__":
    main()
