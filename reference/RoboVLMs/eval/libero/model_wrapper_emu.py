import json
import torch
import numpy as np
from queue import Queue
from PIL import Image

from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("/share/project/yuqi.wang/UniVLA/reference/Emu3")
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids
        self.token_confidences = []  # per-token max prob after masking

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")

        # softmax over allowed tokens → save max prob as confidence for this token
        probs = torch.softmax(scores, dim=-1)
        self.token_confidences.append(probs.max(dim=-1).values.item())
        return scores

class EmuVLAModel:
    # model option
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device

        ## hard code here
        self.window_size = 2
        self.predict_action_frames = 10
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = True
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.use_cot = False  # always disable CoT

        self.video_mode = False

        # SPSA optimization settings
        self.use_spsa = False   # set True to enable L optimization at inference
        self.spsa_n = 20        # number of SPSA iterations
        self.spsa_epsilon = 0.05  # perturbation scale
        self.spsa_alpha = 0.01    # learning rate
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

        self.kwargs = dict(
            mode='VLA',
            padding="longest",
        )
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

    def init_config(self, device):
        
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        # fast tokenization
        fast_path = "/share/project/yuqi.wang/UniVLA/pretrain/fast"
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

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


    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image):
        # preprocess image
        agent_view = image['full_image']
        agent_view = Image.fromarray(agent_view)
        agent_view = agent_view.resize((200, 200))
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None
        if "wrist_image" in image:
            gripper_view = image['wrist_image']
            gripper_view = Image.fromarray(gripper_view)
            gripper_view = gripper_view.resize((200, 200))
            gripper_x = self.image_processor(gripper_view, return_tensors="pt")["pixel_values"].cuda()  
            gripper_code = self.image_tokenizer.encode(gripper_x)

        return (
            image_code,
            gripper_code,
        )

    def step(self, image, goal):
        input_dict = dict()
        
        image_code, gripper_code = self.preprocess(image)

        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 初始化输入ID、token类型ID和attention mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # 遍历历史图像
            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # 更新pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs

        if self.use_fast:
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]

            def _run_generate(inputs_embeds=None, input_ids=None, attn_mask=None):
                """Run one generate() call and return (outputs, chunk_confidence)."""
                proc = ActionIDConstraintLogitsProcessor(allowed_token_ids)
                gen_kwargs = dict(
                    max_new_tokens=80,
                    logits_processor=[proc],
                    attention_mask=attn_mask.to(self.device),
                )
                with torch.no_grad():
                    if inputs_embeds is not None:
                        out = self.model.generate(
                            inputs_embeds=inputs_embeds,
                            **{k: v for k, v in {
                                'pad_token_id': self.GENERATION_CONFIG.pad_token_id,
                                'bos_token_id': self.GENERATION_CONFIG.bos_token_id,
                                'eos_token_id': self.GENERATION_CONFIG.eos_token_id,
                                'do_sample': self.GENERATION_CONFIG.do_sample,
                            }.items()},
                            **gen_kwargs,
                        )
                    else:
                        out = self.model.generate(
                            input_ids.to(self.device),
                            self.GENERATION_CONFIG,
                            **gen_kwargs,
                        )
                conf = float(np.mean(proc.token_confidences)) if proc.token_confidences else 0.0
                return out, conf

            if self.use_spsa:
                # --- SPSA optimization of L in embedding space ---
                hidden_size = self.model.model.embed_tokens.weight.shape[1]  # 4096
                L = torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16, device=self.device)

                # base embeddings (frozen, computed once)
                with torch.no_grad():
                    base_embeds = self.model.model.embed_tokens(
                        final_inputs.input_ids.to(self.device)
                    )  # (1, seq_len, hidden_size)
                base_mask = final_inputs.attention_mask.to(self.device)
                L_mask = torch.ones(1, 1, dtype=base_mask.dtype, device=self.device)
                combined_mask = torch.cat([L_mask, base_mask], dim=1)

                for _ in range(self.spsa_n):
                    # random ±1 perturbation vector (Rademacher)
                    delta = (2 * torch.bernoulli(
                        torch.ones_like(L) * 0.5
                    ) - 1) * self.spsa_epsilon

                    embeds_plus  = torch.cat([L + delta, base_embeds], dim=1)
                    embeds_minus = torch.cat([L - delta, base_embeds], dim=1)

                    _, conf_plus  = _run_generate(inputs_embeds=embeds_plus,  attn_mask=combined_mask)
                    _, conf_minus = _run_generate(inputs_embeds=embeds_minus, attn_mask=combined_mask)

                    # SPSA gradient estimate → gradient ascent (maximize confidence)
                    grad = (conf_plus - conf_minus) / (2.0 * self.spsa_epsilon)
                    L = L + self.spsa_alpha * grad * delta / (delta ** 2 + 1e-8)

                # final generate with optimized L*
                embeds_star = torch.cat([L, base_embeds], dim=1)
                outputs, chunk_confidence = _run_generate(inputs_embeds=embeds_star, attn_mask=combined_mask)
                # outputs contains only generated tokens (no input prefix) when using inputs_embeds
                orig_outputs = outputs
                outputs = outputs[:, :-1]  # omit eoa token
            else:
                # --- Standard Block 3 generate (no SPSA) ---
                outputs, chunk_confidence = _run_generate(
                    input_ids=final_inputs.input_ids,
                    attn_mask=final_inputs.attention_mask,
                )
                orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
                outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]

            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            chunk_confidence = 0.0

        # unnormalize action
        action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action

        if self.use_cot:
            return action_pred, thought, chunk_confidence
        else:
            return action_pred, chunk_confidence
    
    def unormalize_action(self, action):
        action_high = np.array([
            0.93712500009996,
            0.86775000009256,
            0.93712500009996,
            0.13175314309916836,
            0.19275000005139997,
            0.3353504997073735,
            0.9996000000999599
        ])
        action_low = np.array([
            -0.7046250000751599,
            -0.80100000008544,
            -0.9375000001,
            -0.11467779149968735,
            -0.16395000004372,
            -0.2240490058320433,
            -1.0000000001
        ])
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
