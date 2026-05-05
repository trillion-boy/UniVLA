"""
Token-level fovea LoRA fine-tuning of UniVLA (Emu3MoE).

Key difference from train_lora_foveated.py (image-level LoRA):
  - VQ-VAE always sees clean, unmodified pixels  → zero image-level OOD
  - After VQ encoding, a circular bg_token mask is applied to peripheral tokens
  - The model learns: "bg_token in periphery = don't attend, fovea = attend here"

The circular mask is applied by monkey-patching the VQ tokenizer's encode()
inside the dataset. The current episode's fovea center is passed via a
thread-local variable set in a patched __getitem__.

Usage:
    conda run -n univla python train_lora_token_fovea.py \
        --emu-hub      /content/pretrain/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K \
        --vision-hub   /content/pretrain/Emu3-VisionTokenizer \
        --fast-path    /content/UniVLA/pretrain/fast_bridge_t5_s50 \
        --data-pkl     /content/bridge_token_fovea_train.pkl \
        --output-dir   /content/lora_token_fovea \
        --num-epochs   3 \
        --fovea-fraction 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Disable TF/Keras before any transformers import to avoid:
#   - cv2/_ARRAY_API error (keras→cv2 with NumPy 2.x)
#   - Keras 3 / tf-keras incompatibility in transformers.trainer
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import transformers as hf
from transformers import AutoModel, AutoImageProcessor

_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / ".." / ".." / "..").resolve()
_EMU3 = _ROOT / "reference" / "Emu3"
_TRAIN = _ROOT / "train"
for _p in [str(_ROOT), str(_EMU3), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emu3.mllm import Emu3MoE, Emu3Tokenizer  # noqa


# ── Thread-local fovea center (set per dataset item) ──────────────────────────
_fovea_local = threading.local()


def _apply_circular_mask(
    tokens: torch.Tensor,       # (1, H_t, W_t)
    cx_norm: float,
    cy_norm: float,
    fovea_fraction: float,
) -> torch.Tensor:
    H_t, W_t = tokens.shape[-2], tokens.shape[-1]
    cx_t = cx_norm * W_t
    cy_t = cy_norm * H_t
    r    = min(H_t, W_t) * fovea_fraction

    y_idx = torch.arange(H_t, dtype=torch.float32)
    x_idx = torch.arange(W_t, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
    dist = torch.sqrt((xx - cx_t) ** 2 + (yy - cy_t) ** 2)

    peripheral = dist > r
    bg_token   = int(tokens.flatten().mode().values.item())
    tokens_out = tokens.clone()
    tokens_out[0][peripheral] = bg_token
    n_sharp = int((~peripheral).sum())
    return tokens_out


def _patch_dataset_and_tokenizer(vision_hub: str, fovea_fraction: float) -> None:
    """
    Two patches applied together:
    1. vision_hub path fix (same as original train_lora_foveated.py)
    2. circular token mask applied after VQ encoding
    """
    import datasets as ds_module

    original_init    = ds_module.Emu3SFTDataset.__init__
    original_getitem = ds_module.Emu3SFTDataset.__getitem__

    # ── patch __init__: fix vision_hub + store fovea centers ──────────────────
    def patched_init(self, args, tokenizer):
        original_raw = args.raw_image
        args.raw_image = False
        original_init(self, args, tokenizer)
        args.raw_image = original_raw

        if original_raw:
            self.raw_image   = True
            self.vision_hub  = vision_hub
            self.image_processor = AutoImageProcessor.from_pretrained(
                vision_hub, trust_remote_code=True
            )
            # Load VQ tokenizer on GPU
            _vq = AutoModel.from_pretrained(
                vision_hub, trust_remote_code=True
            ).cuda().eval()

            _orig_enc = _vq.encode

            def _masked_encode(pixel_values):
                with torch.no_grad():
                    tokens = _orig_enc(pixel_values.to("cuda")).cpu()  # (1,H_t,W_t)
                fc = getattr(_fovea_local, "value", None)
                if fc is not None:
                    cx_norm, cy_norm = fc
                    tokens = _apply_circular_mask(tokens, cx_norm, cy_norm,
                                                   fovea_fraction)
                return tokens

            _vq.encode = _masked_encode
            self.image_tokenizer = _vq
            self.image_processor.min_pixels = 80 * 80

            # Store fovea centers indexed by dataset position
            self._fovea_centers = []
            if hasattr(args, '_fovea_centers'):
                self._fovea_centers = args._fovea_centers
            print(f"[patch] vision_hub={vision_hub}  "
                  f"fovea_fraction={fovea_fraction}  "
                  f"episodes_with_fovea={len(self._fovea_centers)}")

    # ── patch __getitem__: set thread-local fovea center ──────────────────────
    def patched_getitem(self, index):
        if index < len(self._fovea_centers):
            _fovea_local.value = self._fovea_centers[index]
        else:
            _fovea_local.value = (0.5, 0.5)
        result = original_getitem(self, index)
        _fovea_local.value = None
        return result

    ds_module.Emu3SFTDataset.__init__    = patched_init
    ds_module.Emu3SFTDataset.__getitem__ = patched_getitem


def apply_lora(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emu-hub",        required=True)
    p.add_argument("--vision-hub",     required=True)
    p.add_argument("--fast-path",      required=True)
    p.add_argument("--data-pkl",       required=True)
    p.add_argument("--output-dir",     default="/content/lora_token_fovea")
    p.add_argument("--num-epochs",     type=int,   default=3)
    p.add_argument("--batch-size",     type=int,   default=1)
    p.add_argument("--grad-accum",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--lora-r",         type=int,   default=8)
    p.add_argument("--lora-alpha",     type=int,   default=16)
    p.add_argument("--lora-dropout",   type=float, default=0.05)
    p.add_argument("--fovea-fraction", type=float, default=0.4,
                   help="Fovea circle radius as fraction of token-grid short side")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load pickle + extract fovea centers ───────────────────────────────────
    import pickle
    with open(args.data_pkl, "rb") as f:
        raw_data = pickle.load(f)
    fovea_centers = [ep.get("fovea_center", (0.5, 0.5)) for ep in raw_data]
    print(f"[data] {len(raw_data)} episodes loaded. "
          f"Fovea centers present: {sum(1 for e in raw_data if 'fovea_center' in e)}")

    # ── Patch dataset + tokenizer ──────────────────────────────────────────────
    _patch_dataset_and_tokenizer(args.vision_hub, args.fovea_fraction)
    from datasets import Emu3SFTDataset  # noqa (import after patch)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = Emu3Tokenizer.from_pretrained(
        args.emu_hub, padding_side="right", use_fast=False
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    @dataclass
    class DataArguments:
        data_path:               str   = args.data_pkl
        null_prompt_prob:        float = 0.0
        apply_loss_on_only_vision: bool = False
        apply_loss_on_only_text: bool  = False
        apply_loss_on_only_action: bool = True
        ignore_index:            int   = -100
        visual_token_pattern:    str   = "<|visual token {token_id:0>6d}|>"
        codebook_size:           int   = 32768
        frames:                  int   = 1
        VL:                      bool  = False
        actions:                 bool  = True
        actions_format:          str   = "fast"
        action_frames:           int   = 5
        use_gripper:             bool  = False
        action_tokenizer_path:   str   = args.fast_path
        video_format:            str   = "interleave"
        random_frame_sampling:   bool  = True
        raw_image:               bool  = True
        post_training:           bool  = False
        datasets_weight:         bool  = False
        without_text:            bool  = False
        real_robot:              bool  = False
        with_cot:                bool  = False
        _fovea_centers:          list  = field(default_factory=list)

    data_args = DataArguments(_fovea_centers=fovea_centers)
    train_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    print(f"[dataset] {len(train_dataset)} samples.")

    # ── Model + LoRA ──────────────────────────────────────────────────────────
    print("[init] Loading Emu3MoE ...")
    model = Emu3MoE.from_pretrained(
        args.emu_hub, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = hf.TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = hf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print(f"[train] LoRA fine-tuning with fovea_fraction={args.fovea_fraction} ...")
    trainer.train()

    lora_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(lora_path)
    print(f"\n[done] LoRA adapter saved → {lora_path}")


if __name__ == "__main__":
    main()
