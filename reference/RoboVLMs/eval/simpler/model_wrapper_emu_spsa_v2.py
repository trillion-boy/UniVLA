"""
EmuVLAInferenceSPSA_v2: Task-embedding warm-init + cosine-weighted L injection.

Based on EmuVLAInference (model_wrapper.py) — existing code is untouched.

Changes from vanilla (no-SPSA) EmuVLAInference:
  1. Persistent L for visual token injection, optimized by SPSA.
  2. [v2] L warm-init from task embedding * scale
         → seed L in semantically meaningful direction instead of zeros
  3. [v2] Per-position cosine-similarity weighted L injection
         → concentrate perturbation on task-relevant visual tokens
         → weights are mean-normalized (×N_vis) so total energy matches uniform

Weight normalization rationale:
  softmax(cosine_sim / T) sums to 1 → avg weight = 1/N → energy 1/N of uniform
  × N_vis makes avg weight = 1 → energy identical to uniform on average,
  but task-relevant positions get >1× and irrelevant positions get <1×

Hyperparameter guide:
  task_emb_scale: 0.01 keeps warm-init small; SPSA can freely override it
  cosine_temp:    0.1 = peaked (few tokens dominate),
                  1.0 = soft (near-uniform), tune to task
  spsa_threshold: trigger threshold on rolling_conf (lower = less aggressive)
  spsa_n:         SPSA iterations per step (more = better gradient, slower)
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers.feature_extraction_utils import BatchFeature

from eval.simpler.model_wrapper import EmuVLAInference, ActionIDConstraintLogitsProcessor


class EmuVLAInferenceSPSA_v2(EmuVLAInference):
    """
    SPSA visual-injection with task-embedding warm-init and cosine-weighted apply.

    SPSA pipeline (per step):
        task_emb = embed_tokens(text_ids).mean()          # (hidden_size,)
        if first run: L = task_emb * task_emb_scale       # warm-init
        else:         L = L_persistent                    # warm-start

        for i in range(spsa_n):
            N   = num visual tokens
            w   = softmax(cosine_sim(vis_embs, task_emb) / T) * N  # (N,), mean=1
            f±  = conf(vis_embs + (L ± δ) * w)
            g   = (f+ - f-) / 2ε
            L  += α * g * δ / (δ² + 1e-8)
            L   = clip_norm(L, max_norm)

        L_persistent = β * L
        action = generate(vis_embs + L * w)
    """

    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        policy_setup="widowx_bridge",
        # SPSA on/off
        use_spsa=True,
        # SPSA hyperparams
        spsa_n=10,
        spsa_epsilon=0.05,
        spsa_alpha=0.01,
        # warm-start decay: L_persistent = beta * L_star
        spsa_beta=0.7,
        # L2-norm clipping
        spsa_max_norm=1.0,
        # score-triggered activation (fires when rolling_score < threshold)
        spsa_threshold=0.55,
        # EMA decay for rolling_score
        ema_decay=0.3,
        # [v2] warm-init: L_init = task_emb * this_scale
        task_emb_scale=0.01,
        # [v2] cosine weight temperature (lower = more peaked)
        cosine_temp=0.1,
        # [v3] SPSA objective weights: score = alpha*visual_grounding + beta*task_alignment
        score_alpha=0.5,
        score_beta=0.5,
        # [v2] debug: print weight stats once per N steps
        debug_weight_every=0,   # 0 = disabled
    ):
        super().__init__(emu_hub, vq_hub, vision_hub, device, policy_setup)

        self.use_spsa = use_spsa
        self.spsa_n = spsa_n
        self.spsa_epsilon = spsa_epsilon
        self.spsa_alpha = spsa_alpha
        self.spsa_beta = spsa_beta
        self.spsa_max_norm = spsa_max_norm
        self.spsa_threshold = spsa_threshold
        self.ema_decay = ema_decay
        self.task_emb_scale = task_emb_scale
        self.cosine_temp = cosine_temp
        self.score_alpha = score_alpha
        self.score_beta = score_beta
        self.debug_weight_every = debug_weight_every

        hidden_size = self.model.model.embed_tokens.weight.shape[1]
        self.L_persistent = torch.zeros(hidden_size, dtype=torch.bfloat16, device=self.device)
        self.rolling_score = 1.0  # start optimistic → SPSA won't fire immediately
        self.last_combined_score = 0.0
        self.last_confidence = 0.0   # raw conf, kept for logging only
        self._spsa_ever_ran = False
        self._step_count = 0
        # Check if last-layer hook is possible (LLaMA-style architecture)
        self._has_layers = (
            hasattr(self.model, 'model')
            and hasattr(self.model.model, 'layers')
            and len(self.model.model.layers) > 0
        )

    # ------------------------------------------------------------------
    # Reset: called at episode start
    # ------------------------------------------------------------------

    def reset(self):
        super().reset()
        hidden_size = self.model.model.embed_tokens.weight.shape[1]
        self.L_persistent = torch.zeros(hidden_size, dtype=torch.bfloat16, device=self.device)
        self.rolling_score = 1.0
        self.last_combined_score = 0.0
        self.last_confidence = 0.0
        self._spsa_ever_ran = False
        self._step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_visual_mask(self, input_ids_1d):
        """Bool mask: True at positions strictly between boi_token and eoi_token."""
        boi_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.boi_token)
        eoi_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eoi_token)

        visual_mask = torch.zeros(len(input_ids_1d), dtype=torch.bool, device=self.device)
        boi_pos = (input_ids_1d == boi_id).nonzero(as_tuple=True)[0]
        eoi_pos = (input_ids_1d == eoi_id).nonzero(as_tuple=True)[0]

        n_pairs = min(len(boi_pos), len(eoi_pos))
        for i in range(n_pairs):
            s, e = boi_pos[i].item(), eoi_pos[i].item()
            if s < e:
                visual_mask[s + 1:e] = True
        return visual_mask

    @torch.no_grad()
    def _compute_task_emb(self, text_input_ids):
        """
        Mean-pool embed_tokens over text token positions → (hidden_size,).

        Args:
            text_input_ids: (1, text_len) LongTensor (text tokens only, no image)
        Returns:
            task_emb: (hidden_size,) bfloat16 tensor on self.device
        """
        text_ids = text_input_ids.to(self.device)              # (1, text_len)
        embs = self.model.model.embed_tokens(text_ids)         # (1, text_len, H)
        return embs.mean(dim=1)[0]                             # (H,)

    def _cosine_weights(self, vis_embs, task_emb):
        """
        Per-position cosine similarity weights, mean-normalized to 1.

        Args:
            vis_embs: (N, hidden_size) float32 tensor
            task_emb: (hidden_size,)   float32 tensor

        Returns:
            weights: (N,) float32, mean ≈ 1, sum = N
        """
        N = vis_embs.shape[0]
        vis_norm  = F.normalize(vis_embs, dim=1)                      # (N, H)
        task_norm = F.normalize(task_emb.unsqueeze(0), dim=1)         # (1, H)
        cosine_sim = (vis_norm @ task_norm.T).squeeze(1)              # (N,)
        weights = F.softmax(cosine_sim / self.cosine_temp, dim=0) * N # (N,), mean=1
        return weights

    # ------------------------------------------------------------------
    # Main inference step (override parent)
    # ------------------------------------------------------------------

    def step(self, image, goal):
        self._step_count += 1

        # ---------- Preprocessing (identical to parent) ----------
        image_code, gripper_code = self.preprocess(image)
        prompt = goal

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        kwargs = dict(mode='VLA_Video', padding="longest")
        pos_inputs = self.processor.video_process(
            text=prompt, video_tokens=video_code, gripper_tokens=gripper_code,
            context_frames=self.context_frames, frames=self.predict_frames,
            return_tensors="pt", **kwargs,
        )

        self.add_image(pos_inputs)
        history = self.get_history()
        action_history = self.get_action_history()

        all_input_ids, all_token_type_ids, all_attention_mask = [], [], []
        all_input_ids.append(text_tokens['input_ids'])
        all_token_type_ids.append(text_tokens['token_type_ids'])
        all_attention_mask.append(text_tokens['attention_mask'])

        for i in range(len(history)):
            img_ids   = history[i]['input_ids']
            img_ttids = history[i]['token_type_ids']
            img_amask = history[i]['attention_mask']

            if i < len(action_history):
                act_ids = action_history[i]
                all_input_ids.extend([img_ids, act_ids])
                all_token_type_ids.extend([img_ttids, torch.zeros_like(act_ids)])
                all_attention_mask.extend([img_amask, torch.ones_like(act_ids)])
            else:
                all_input_ids.append(img_ids)
                all_token_type_ids.append(img_ttids)
                all_attention_mask.append(img_amask)

        final_inputs = pos_inputs.copy()
        final_inputs['input_ids']      = torch.cat(all_input_ids,      dim=1)
        final_inputs['token_type_ids'] = torch.cat(all_token_type_ids, dim=1)
        final_inputs['attention_mask'] = torch.cat(all_attention_mask, dim=1)

        # ---------- [v2] Compute task embedding from text tokens only ----------
        task_emb = self._compute_task_emb(text_tokens['input_ids'])  # (H,) bf16

        # ---------- Generate setup ----------
        last_token_id = self.tokenizer.pad_token_id - 1
        allowed_token_ids = list(range(
            last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1
        )) + [self.eoa_token_id]

        seq_len       = final_inputs.input_ids.shape[1]
        input_ids_dev = final_inputs.input_ids.to(self.device)
        base_mask     = final_inputs.attention_mask.to(self.device)
        visual_mask   = self._get_visual_mask(final_inputs.input_ids[0].to(self.device))
        N_vis         = int(visual_mask.sum().item())

        # Pre-compute task_emb in float32 for cosine computation
        task_emb_f32 = task_emb.float()

        def _run_generate(L_vec):
            """
            Generate with [v2] cosine-weighted L injection via forward hook.

            Returns: (gen, orig_gen, combined_score, raw_conf)
              combined_score = score_alpha * visual_grounding_norm
                             + score_beta  * task_alignment_norm
              raw_conf       = mean max-prob over action tokens (original conf)

            visual_grounding: cosine similarity between each generated action token's
              last-layer hidden state and the visual token hidden states from the
              same layer during the prompt pass. High = model attends to scene.
            task_alignment: cosine similarity between action hidden state and
              task embedding. High = action is grounded in task instruction.
            """
            # ---- buffers for scoring ----
            _vis_hidden_ref = [None]   # visual token hidden states from prompt pass
            _task_align_buf = []
            _vis_ground_buf = []

            def _hook_embed(module, input, output):
                """Inject L into visual token embeddings (prompt pass only)."""
                if (
                    L_vec is not None
                    and output.shape[1] == seq_len
                    and N_vis > 0
                ):
                    output = output.clone()
                    vis_embs = output[0, visual_mask, :]       # (N_vis, H) bf16

                    weights = self._cosine_weights(
                        vis_embs.float(), task_emb_f32
                    )  # (N_vis,) float32, mean=1

                    if (
                        self.debug_weight_every > 0
                        and self._step_count % self.debug_weight_every == 0
                    ):
                        w = weights.cpu()
                        print(
                            f"[SPSA-v2] weight stats: "
                            f"max={w.max():.3f} min={w.min():.3f} "
                            f"std={w.std():.3f} top5={w.topk(5).values.tolist()}"
                        )

                    w_bf16 = weights.to(L_vec.dtype).unsqueeze(1)  # (N_vis, 1)
                    output[0, visual_mask, :] = vis_embs + L_vec * w_bf16

                return output

            def _hook_last_layer(module, inp, output):
                """
                Capture last-layer hidden states for grounding/alignment scoring.
                - Prompt pass  (cur_len == seq_len): save visual token hiddens as ref.
                - Action steps (cur_len  > seq_len): compute task_align + vis_ground.
                """
                hidden = output[0]           # (1, cur_len, H)
                cur_len = hidden.shape[1]

                if cur_len == seq_len:
                    # Prompt pass — save visual token hidden states
                    if N_vis > 0:
                        _vis_hidden_ref[0] = hidden[0, visual_mask, :].detach().float()
                else:
                    # Action generation step
                    last_h = hidden[0, -1, :].float()   # (H,)

                    # Task-action alignment
                    t_align = F.cosine_similarity(
                        last_h.unsqueeze(0), task_emb_f32.unsqueeze(0)
                    ).item()
                    _task_align_buf.append(t_align)

                    # Visual grounding: sim(action_hidden, visual_hiddens from last layer)
                    if _vis_hidden_ref[0] is not None:
                        vis_h = _vis_hidden_ref[0]          # (N_vis, H)
                        lh_n  = F.normalize(last_h.unsqueeze(0), dim=1)   # (1, H)
                        vn    = F.normalize(vis_h, dim=1)                  # (N_vis, H)
                        g = (lh_n @ vn.T).mean().item()    # scalar in [-1, 1]
                        _vis_ground_buf.append(g)

            handle_embed = self.model.model.embed_tokens.register_forward_hook(_hook_embed)
            handle_last  = (
                self.model.model.layers[-1].register_forward_hook(_hook_last_layer)
                if self._has_layers else None
            )
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
                handle_embed.remove()
                if handle_last is not None:
                    handle_last.remove()

            raw_conf = float(np.mean(proc.token_confidences)) if proc.token_confidences else 0.0
            orig_gen = out[:, seq_len:]
            gen      = out[:, seq_len:-1]

            # ---- compute combined score ----
            if _task_align_buf and _vis_ground_buf:
                task_align_n = (float(np.mean(_task_align_buf)) + 1) / 2   # [-1,1] → [0,1]
                vis_ground_n = (float(np.mean(_vis_ground_buf)) + 1) / 2
                combined_score = (
                    self.score_alpha * vis_ground_n
                    + self.score_beta  * task_align_n
                )
            else:
                # Fallback: architecture doesn't expose layers, use raw conf
                combined_score = raw_conf

            return gen, orig_gen, combined_score, raw_conf

        # ---------- SPSA or warm-start apply ----------
        if self.use_spsa and self.rolling_score < self.spsa_threshold:
            # [v2] Warm-init: seed L from task embedding on first SPSA run
            if not self._spsa_ever_ran:
                L = (task_emb * self.task_emb_scale).to(torch.bfloat16)
                print(
                    f"[SPSA-v2] warm-init: L_norm={L.norm():.4f} "
                    f"task_emb_norm={task_emb.norm():.4f}"
                )
            else:
                L = self.L_persistent.clone()

            for _ in range(self.spsa_n):
                # Rademacher ±ε perturbation
                delta = (2 * torch.bernoulli(torch.ones_like(L) * 0.5) - 1) * self.spsa_epsilon

                _, _, score_plus,  _ = _run_generate(L + delta)
                _, _, score_minus, _ = _run_generate(L - delta)

                # SPSA gradient ascent (maximize combined_score)
                grad = (score_plus - score_minus) / (2.0 * self.spsa_epsilon)
                L = L + self.spsa_alpha * grad * delta / (delta ** 2 + 1e-8)

                # L2-norm clipping
                L_norm = torch.norm(L)
                if L_norm > self.spsa_max_norm:
                    L = L * self.spsa_max_norm / L_norm

            self._spsa_ever_ran = True

            outputs, orig_outputs, chunk_score, chunk_confidence = _run_generate(L)

            # Warm-start for next chunk: quality-gate by combined_score
            # High score → preserve L more; Low score → trust less
            quality_gate = min(chunk_score / max(self.spsa_threshold, 1e-6), 1.0)
            self.L_persistent = (self.spsa_beta * L * quality_gate).detach()
        else:
            # Apply persistent L from previous chunk (with cosine weighting)
            outputs, orig_outputs, chunk_score, chunk_confidence = _run_generate(self.L_persistent)

        # EMA update on rolling_score (used for SPSA trigger)
        self.rolling_score = (
            self.ema_decay * self.rolling_score
            + (1 - self.ema_decay) * chunk_score
        )
        self.last_combined_score = chunk_score
        self.last_confidence = chunk_confidence  # raw conf, for logging only

        # ---------- Decode action (identical to parent) ----------
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

        action = self.unormalize_action(action)
        if self.use_one_step:
            action_pred = action[0:1]
        else:
            action_pred = action

        res = [self.transform_action(action[[i], :]) for i in range(action.shape[0])]
        raw_actions = [_[0] for _ in res]
        env_actions = [_[1] for _ in res]

        return raw_actions, env_actions
