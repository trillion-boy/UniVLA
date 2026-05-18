"""
latent_saccade_postnorm.py

LatentSaccadePostNormEmuVLAInference:
  Identical to LatentSaccadeEmuVLAInference (embed_tokens hook) except
  the spatial weight mask is applied AFTER input_layernorm (RMSNorm) in
  every Emu3 decoder layer.

Why the change matters
----------------------
embed_tokens hook (original):
  token IDs → embed_tokens × weight → RMSNorm → Q,K,V
                                         ↑ weight cancelled here

post-RMSNorm hook (this file):
  token IDs → embed_tokens → RMSNorm × weight → Q,K,V
                                         ↑ weight survives into Q,K,V

This mirrors the approach used in latent_saccade_tracevla.py.

Recommended weights (fovea-only boost, Method 1)
------------------------------------------------
  bg_weight=1.0        — do NOT suppress bg (UniVLA has 93% bg tokens;
                         suppressing them destroys spatial planning)
  place_src_weight=1.1 — mild boost for source/dest object region
  fovea_weight=1.3     — boost target object region
                         → fovea attention score: 1.3² = 1.69× amplified
                         → bg attention score: unchanged

Usage
-----
  from latent_saccade_postnorm import LatentSaccadePostNormEmuVLAInference

  model = LatentSaccadePostNormEmuVLAInference(
      emu_hub=..., vq_hub=..., vision_hub=..., device="cuda",
      fast_path=...,
      bg_weight=1.0, place_src_weight=1.1, fovea_weight=1.3,
      ...
  )
  # then use exactly like LatentSaccadeEmuVLAInference
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from foveated_inference import (
    GroundingDINOWrapper,
    LatentSaccadeEmuVLAInference,
    ActionIDConstraintLogitsProcessor,
)


class LatentSaccadePostNormEmuVLAInference(LatentSaccadeEmuVLAInference):
    """
    LatentSaccade with post-RMSNorm weight masking for Emu3 (UniVLA).

    Changes vs. parent
    ------------------
    - __init__  : registers persistent forward hooks on every decoder layer's
                  input_layernorm instead of a per-call embed_tokens hook.
    - step()    : builds a full-sequence weight tensor and stores it in
                  self._current_seq_weight before generate(); the layernorm
                  hooks read it and clear it afterward.
    - __del__   : removes all registered hook handles.

    Weight tensor
    -------------
    Shape: (seq_len,)   — 1.0 for all non-visual positions,
                          spatial weight for visual token positions in
                          the current frame.
    The layernorm hook multiplies hidden_states by w.view(1, seq_len, 1).
    """

    def __init__(self, **kwargs):
        self._current_seq_weight: Optional[torch.Tensor] = None
        self._ln_hook_handles: List = []

        super().__init__(**kwargs)

        self._register_postnorm_hooks()

    # ── Layer discovery ───────────────────────────────────────────────────

    def _find_decoder_layers(self):
        """Return the list of decoder layers from Emu3MoE."""
        candidates = [
            lambda m: m.model.layers,
            lambda m: m.language_model.model.layers,
            lambda m: m.transformer.h,
        ]
        for fn in candidates:
            try:
                layers = fn(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(
            "[PostNorm] Cannot find decoder layers in model. "
            "Tried model.model.layers, model.language_model.model.layers, "
            "model.transformer.h"
        )

    def _find_layernorm(self, layer):
        """Return the input_layernorm sub-module of a decoder layer."""
        for attr in ("input_layernorm", "ln_1", "layer_norm_1", "norm1"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise RuntimeError(
            f"[PostNorm] Cannot find input_layernorm in layer {type(layer).__name__}. "
            f"Attributes: {[a for a in dir(layer) if 'norm' in a.lower() or 'ln' in a.lower()]}"
        )

    # ── Hook registration ─────────────────────────────────────────────────

    def _register_postnorm_hooks(self):
        layers = self._find_decoder_layers()

        for layer in layers:
            ln = self._find_layernorm(layer)

            def _make_hook(self_ref):
                def _hook(module, inp, output):
                    if not self_ref._enable_latent_mask:
                        return output
                    if self_ref._current_seq_weight is None:
                        return output
                    # Skip single-token autoregressive generation steps
                    if output.shape[1] <= 1:
                        return output
                    w = self_ref._current_seq_weight.to(
                        dtype=output.dtype, device=output.device
                    )
                    # Clamp in case seq lengths differ (safety)
                    seq_len = output.shape[1]
                    w = w[:seq_len]
                    out = output.clone()
                    out = out * w.view(1, seq_len, 1)
                    return out
                return _hook

            handle = ln.register_forward_hook(_make_hook(self))
            self._ln_hook_handles.append(handle)

        print(
            f"[PostNorm] Registered post-RMSNorm hooks on "
            f"{len(self._ln_hook_handles)} decoder layers"
        )

    # ── Sequence weight builder ───────────────────────────────────────────

    def _build_seq_weight(
        self,
        input_ids: torch.Tensor,   # (1, seq_len)
        weight_1d: Optional[torch.Tensor],  # (H_t * W_t,) or None
        frame_start: int,
    ) -> Optional[torch.Tensor]:
        """
        Build a (seq_len,) weight tensor:
          1.0  for all positions that are not visual tokens in the current frame
          w_i  for visual token positions in the current frame

        Returns None if weight_1d is None (disables masking).
        """
        if weight_1d is None:
            return None

        seq_len = input_ids.shape[1]
        seq_weight = torch.ones(seq_len, dtype=torch.float32)

        frame_ids = input_ids[0, frame_start:]
        is_visual = (frame_ids >= self.vis_start) & (frame_ids <= self.vis_end)
        vis_idx = is_visual.nonzero(as_tuple=True)[0]   # positions within frame
        n_vis = vis_idx.numel()

        if n_vis > 0:
            w = weight_1d[:n_vis]
            abs_idx = frame_start + vis_idx
            seq_weight[abs_idx] = w

        return seq_weight

    # ── step: override to use postnorm hooks instead of embed hook ────────

    def step(self, image: np.ndarray, goal: str):
        # ── 1. Sync instruction → saccade nouns ───────────────────────────
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINOWrapper.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun   = dst
            print(f"[PostNorm] Instruction → src='{src}'  dst='{dst}'")

        # ── 2. DINO detection → spatial weight map ────────────────────────
        fovea_bbox, secondary_bbox = self._get_bboxes(image)
        weight_1d = self._build_weight_map(image, fovea_bbox, secondary_bbox)

        # ── 3. Encode image cleanly (no pixel modification) ───────────────
        image_code, gripper_code = self.preprocess(image)

        # ── 4. Build full input sequence (identical to parent) ────────────
        prompt     = goal
        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type="pt")

        pos_inputs = self.processor.video_process(
            text=prompt,
            video_tokens=video_code,
            gripper_tokens=gripper_code,
            context_frames=self.context_frames,
            frames=self.predict_frames,
            return_tensors="pt",
            mode="VLA_Video",
            padding="longest",
        )

        if self.video_mode:
            self.add_image(pos_inputs)
            history        = self.get_history()
            action_history = self.get_action_history()

            all_input_ids      = [text_tokens["input_ids"]]
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
            final_inputs["input_ids"]      = torch.cat(all_input_ids,      dim=1)
            final_inputs["token_type_ids"] = torch.cat(all_token_type_ids, dim=1)
            final_inputs["attention_mask"] = torch.cat(all_attention_mask, dim=1)
        else:
            final_inputs = pos_inputs

        # ── 5. Locate current frame in the full sequence ──────────────────
        current_frame_len = pos_inputs["input_ids"].shape[1]
        frame_start       = final_inputs["input_ids"].shape[1] - current_frame_len

        n_fovea = int((weight_1d >= self._fovea_weight).sum())          if weight_1d is not None else 0
        n_src   = int(((weight_1d >= self._place_src_weight) & (weight_1d < self._fovea_weight)).sum()) if weight_1d is not None else 0
        n_bg    = int((weight_1d < self._place_src_weight).sum())        if weight_1d is not None else 0
        print(
            f"[PostNorm] phase={self.saccade.state}  "
            f"target='{self.saccade.current_target}'  "
            f"fovea={n_fovea}  src={n_src}  bg={n_bg}  "
            f"fovea_bbox={fovea_bbox}"
        )

        # ── 6. Build seq_weight → generate (postnorm hooks fire) ──────────
        self._current_seq_weight = self._build_seq_weight(
            final_inputs["input_ids"], weight_1d, frame_start
        )

        last_token_id = self.tokenizer.pad_token_id - 1
        allowed = list(
            range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)
        ) + [self.eoa_token_id]
        action_id_processor = ActionIDConstraintLogitsProcessor(allowed)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=100,
                    logits_processor=[action_id_processor],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )
        finally:
            self._current_seq_weight = None   # always clear after generate

        # ── 7. Decode actions (same as parent) ────────────────────────────
        orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
        outputs      = orig_outputs[:, :-1]
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
        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        raw_actions, env_actions = [r[0] for r in res], [r[1] for r in res]

        # ── 8. Update saccade from gripper output ─────────────────────────
        if env_actions:
            g = float(np.asarray(env_actions[-1].get("gripper", [1.0])).flat[0])
            gripper_norm = (1.0 - g) / 2.0    # +1=open→0, −1=close→1
            print(
                f"[PostNorm-dbg] g={g:.2f} gripper_norm={gripper_norm:.2f} "
                f"close_count={self.saccade._close_count} "
                f"grasp_steps={self.saccade._grasp_steps}",
                flush=True,
            )
            transitioned = self.saccade.update(gripper_norm)
            if transitioned:
                self._fovea_bbox_cache    = None
                self._secondary_bbox_cache = None
                self._cache_step          = 0

        return raw_actions, env_actions

    # ── Cleanup ───────────────────────────────────────────────────────────

    def __del__(self):
        for handle in getattr(self, "_ln_hook_handles", []):
            handle.remove()
