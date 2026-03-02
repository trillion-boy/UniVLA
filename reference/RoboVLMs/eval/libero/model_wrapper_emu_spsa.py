"""
EmuVLAModelSPSA: ControlMLLM-style visual broadcast injection + warm-start SPSA.

Key differences from EmuVLAModel (model_wrapper_emu.py):
  1. L is broadcast to all visual token positions (ControlMLLM p_v + e_v style)
     instead of prepended as a new token → seq_len unchanged, no mask modification.
  2. L is persistent across chunks within one episode (warm-start).
  3. SPSA fires only when rolling_conf < adaptive_threshold (compute-efficient).
  4. Adaptive threshold decays slightly toward episode end (more aggressive late).
"""

import torch
import numpy as np
from transformers.feature_extraction_utils import BatchFeature

from model_wrapper_emu import EmuVLAModel, ActionIDConstraintLogitsProcessor


class EmuVLAModelSPSA(EmuVLAModel):
    """
    EmuVLAModel with ControlMLLM-style visual broadcast injection and warm-start SPSA.

    SPSA pipeline (training-free, model frozen):
        for i in range(spsa_n):
            embeds± = apply_L(base_embeds, L ± delta, visual_mask)   # visual broadcast
            conf±   = generate(embeds±) → confidence
            grad    = (conf+ - conf-) / (2ε)
            L       = L + α * grad * delta / (delta² + 1e-8)
        L_persistent = β * L*           # warm-start for next chunk
        action = generate(apply_L(base_embeds, L*, visual_mask))

    Trigger: fires only when rolling_conf < adaptive_threshold.
    """

    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        fast_hub=None,
        # SPSA on/off
        use_spsa=True,
        # SPSA hyperparams
        spsa_n=20,
        spsa_epsilon=0.05,
        spsa_alpha=0.01,
        # Warm-start: L_persistent = beta * L_star  (0 = reset every chunk, 1 = no decay)
        spsa_beta=0.7,
        # L2-norm clipping: keeps L within distribution (~1% of typical embedding norm)
        spsa_max_norm=1.0,
        # Confidence-triggered activation
        spsa_threshold=0.60,   # fire SPSA when rolling_conf < threshold
        spsa_ep_len=300,       # episode length for adaptive threshold decay
        ema_decay=0.3,         # EMA smoothing of rolling confidence
    ):
        super().__init__(emu_hub, vq_hub, vision_hub, device, fast_hub)

        self.use_spsa = use_spsa
        self.spsa_n = spsa_n
        self.spsa_epsilon = spsa_epsilon
        self.spsa_alpha = spsa_alpha
        self.spsa_beta = spsa_beta
        self.spsa_max_norm = spsa_max_norm
        self.spsa_threshold = spsa_threshold
        self.spsa_ep_len = spsa_ep_len
        self.ema_decay = ema_decay

        # Episode-level persistent state (reset in reset())
        hidden_size = self.model.model.embed_tokens.weight.shape[1]
        self.L_persistent = torch.zeros(hidden_size, dtype=torch.bfloat16, device=self.device)
        self.rolling_conf = 1.0  # start optimistic so SPSA doesn't fire immediately

    # ------------------------------------------------------------------
    # Reset: called at the start of each episode
    # ------------------------------------------------------------------

    def reset(self):
        super().reset()
        hidden_size = self.model.model.embed_tokens.weight.shape[1]
        self.L_persistent = torch.zeros(hidden_size, dtype=torch.bfloat16, device=self.device)
        self.rolling_conf = 1.0

    # ------------------------------------------------------------------
    # Visual token detection (ControlMLLM: only modify e_v, not e_text)
    # ------------------------------------------------------------------

    def _get_visual_mask(self, input_ids_1d):
        """
        Return a bool mask identifying visual token positions.

        Emu3's token_type_ids are all zeros (tiktoken base doesn't set them),
        and modeling_emu3.py doesn't use token_type_ids in the forward pass.
        We detect visual regions by finding positions strictly between
        boi_token and eoi_token in the sequence.

        Args:
            input_ids_1d: (seq_len,) LongTensor on self.device

        Returns:
            visual_mask: (seq_len,) BoolTensor, True = visual token position
        """
        boi_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.boi_token)
        eoi_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eoi_token)

        visual_mask = torch.zeros(len(input_ids_1d), dtype=torch.bool, device=self.device)
        boi_pos = (input_ids_1d == boi_id).nonzero(as_tuple=True)[0]
        eoi_pos = (input_ids_1d == eoi_id).nonzero(as_tuple=True)[0]

        n_pairs = min(len(boi_pos), len(eoi_pos))
        for i in range(n_pairs):
            s, e = boi_pos[i].item(), eoi_pos[i].item()
            if s < e:
                visual_mask[s + 1:e] = True  # tokens strictly between boi and eoi

        return visual_mask

    # ------------------------------------------------------------------
    # L injection (ControlMLLM p_v + e_v style, broadcast)
    # ------------------------------------------------------------------

    def _apply_L(self, base_embeds, L, visual_mask):
        """
        Broadcast L to all visual token positions (additive).

        ControlMLLM: e_v_new = e_v + p_v
        Here: e_v[:, visual_positions, :] += L  (same L vector for all positions)

        Args:
            base_embeds: (1, seq_len, hidden_size) — frozen embeddings
            L: (hidden_size,) — learnable latent variable
            visual_mask: (seq_len,) bool — visual positions

        Returns:
            embeds: (1, seq_len, hidden_size) with L added at visual positions
        """
        embeds = base_embeds.clone()
        if visual_mask.any():
            embeds[0, visual_mask, :] = embeds[0, visual_mask, :] + L
        return embeds

    # ------------------------------------------------------------------
    # Main inference step (override parent)
    # ------------------------------------------------------------------

    def step(self, image, goal, current_step=0):
        """
        Same as EmuVLAModel.step() but with ControlMLLM-style visual broadcast
        injection and confidence-triggered warm-start SPSA.

        Args:
            image: observation dict with 'full_image' (and optionally 'wrist_image')
            goal: task description string
            current_step: robot step count within this episode (for adaptive threshold)

        Returns:
            action_pred: np.ndarray of predicted actions
            chunk_confidence: float confidence for this chunk
        """
        # ---------- Preprocessing (identical to parent) ----------
        image_code, gripper_code = self.preprocess(image)
        prompt = goal

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(mode='VLA_Video', padding="longest")
            pos_inputs = self.processor.video_process(
                text=prompt, video_tokens=video_code, gripper_tokens=gripper_code,
                context_frames=self.context_frames, frames=self.predict_frames,
                return_tensors="pt", **kwargs,
            )
        else:
            pos_inputs = self.processor.video_process(
                text=prompt, video_tokens=video_code, gripper_tokens=gripper_code,
                context_frames=self.context_frames, frames=self.predict_frames,
                return_tensors="pt", **self.kwargs,
            )

        if self.video_mode:
            self.add_image(pos_inputs)
            history = self.get_history()
            action_history = self.get_action_history()

            all_input_ids, all_token_type_ids, all_attention_mask = [], [], []

            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']

                if i < len(action_history):
                    act_input_ids = action_history[i]
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)

            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = torch.cat(all_input_ids, dim=1)
            final_inputs['token_type_ids'] = torch.cat(all_token_type_ids, dim=1)
            final_inputs['attention_mask'] = torch.cat(all_attention_mask, dim=1)
        else:
            final_inputs = pos_inputs

        # ---------- Fast-tokenizer inference ----------
        if self.use_fast:
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(
                last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1
            )) + [self.eoa_token_id]

            seq_len = final_inputs.input_ids.shape[1]
            input_ids_dev = final_inputs.input_ids.to(self.device)
            base_mask = final_inputs.attention_mask.to(self.device)
            visual_mask = self._get_visual_mask(final_inputs.input_ids[0].to(self.device))

            def _run_generate(L_vec):
                """Generate using input_ids path (identical to baseline) with L injected
                via a forward hook on embed_tokens. Avoids inputs_embeds path which causes
                Emu3MoE to generate EOA immediately and return empty action tensors.
                Uses register_forward_hook instead of module replacement to stay compatible
                with PyTorch's nn.Module attribute restrictions."""

                def _hook_fn(module, input, output):
                    # Add L only to the initial prompt embedding (seq_len positions),
                    # not to the single-token calls during autoregressive generation.
                    if L_vec is not None and output.shape[1] == seq_len and visual_mask.any():
                        output = output.clone()
                        output[0, visual_mask, :] = output[0, visual_mask, :] + L_vec
                    return output

                handle = self.model.model.embed_tokens.register_forward_hook(_hook_fn)
                try:
                    proc = ActionIDConstraintLogitsProcessor(allowed_token_ids)
                    with torch.no_grad():
                        out = self.model.generate(
                            input_ids_dev,
                            self.GENERATION_CONFIG,
                            max_new_tokens=80,
                            logits_processor=[proc],
                            attention_mask=base_mask,
                        )
                finally:
                    handle.remove()

                conf = float(np.mean(proc.token_confidences)) if proc.token_confidences else 0.0
                # Same output parsing as baseline no-SPSA path
                orig_gen = out[:, seq_len:]       # generated tokens (incl. eoa)
                gen = out[:, seq_len:-1]          # generated tokens (excl. eoa)
                return gen, orig_gen, conf

            # Adaptive threshold: starts at spsa_threshold, decreases by 0.05 at ep_len
            t_ratio = min(current_step / max(self.spsa_ep_len, 1), 1.0)
            adaptive_threshold = self.spsa_threshold - 0.05 * t_ratio

            if self.use_spsa and self.rolling_conf < adaptive_threshold:
                # ------ SPSA: optimize L via embed_tokens injection ------
                L = self.L_persistent.clone()

                for _ in range(self.spsa_n):
                    # Rademacher ±ε perturbation
                    delta = (
                        2 * torch.bernoulli(torch.ones_like(L) * 0.5) - 1
                    ) * self.spsa_epsilon

                    _, _, conf_plus  = _run_generate(L + delta)
                    _, _, conf_minus = _run_generate(L - delta)

                    # SPSA gradient ascent (maximize confidence)
                    grad = (conf_plus - conf_minus) / (2.0 * self.spsa_epsilon)
                    L = L + self.spsa_alpha * grad * delta / (delta ** 2 + 1e-8)

                    # L2-norm clipping: prevent distribution shift
                    L_norm = torch.norm(L)
                    if L_norm > self.spsa_max_norm:
                        L = L * self.spsa_max_norm / L_norm

                # Update persistent L with decay (warm-start for next chunk)
                self.L_persistent = (self.spsa_beta * L).detach()

                outputs, orig_outputs, chunk_confidence = _run_generate(L)
            else:
                # ------ No SPSA: apply persistent L from previous chunk ------
                outputs, orig_outputs, chunk_confidence = _run_generate(self.L_persistent)

            # Update rolling confidence (EMA)
            self.rolling_conf = (
                self.ema_decay * self.rolling_conf + (1 - self.ema_decay) * chunk_confidence
            )

            last_token_id_tensor = torch.tensor(
                last_token_id, dtype=outputs.dtype, device=outputs.device
            )
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs,
                time_horizon=self.predict_action_frames,
                action_dim=self.action_dim,
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            chunk_confidence = 0.0

        # ---------- Post-process action (identical to parent) ----------
        action = self.unormalize_action(action)
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        if self.use_one_step:
            action_pred = action[0:1]
        else:
            action_pred = action

        if self.use_cot:
            return action_pred, None, chunk_confidence
        else:
            return action_pred, chunk_confidence
