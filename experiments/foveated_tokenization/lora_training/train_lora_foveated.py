"""
LoRA fine-tuning of UniVLA (Emu3MoE) on foveated Bridge images.

What this does:
  - Loads pretrained Emu3MoE (frozen, ~7B params)
  - Attaches LoRA adapters to attention q_proj / v_proj (trainable, ~tens of MB)
  - Trains on (foveated_image, instruction) → action tokens
  - Loss: cross-entropy on action token predictions only
  - Saves LoRA adapter weights (NOT the full model)

Usage:
    conda run -n univla python train_lora_foveated.py \
        --emu-hub      /content/pretrain/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K \
        --vision-hub   /content/pretrain/Emu3-VisionTokenizer \
        --fast-path    /content/UniVLA/pretrain/fast_bridge_t5_s50 \
        --data-pkl     /local-scratch/bridge_foveated_train.pkl \
        --output-dir   /content/lora_foveated \
        --num-epochs   3 \
        --batch-size   1 \
        --lora-r       8
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import transformers as hf
from transformers import AutoModel, AutoImageProcessor, AutoProcessor

# ── Repo path setup ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / ".." / ".." / "..").resolve()
_EMU3 = _ROOT / "reference" / "Emu3"
_TRAIN = _ROOT / "train"
for _p in [str(_ROOT), str(_EMU3), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emu3.mllm import Emu3MoEConfig, Emu3MoE, Emu3Tokenizer  # noqa: E402


# ── Patch Emu3SFTDataset: fix hardcoded vision_hub path ───────────────────────

def _patch_dataset_vision_hub(vision_hub: str) -> None:
    """
    Emu3SFTDataset hardcodes vision_hub to /share/project/... when raw_image=True.
    Fix: temporarily set raw_image=False so original_init skips the hardcoded
    path, then manually set up image_processor/tokenizer with the correct path.
    """
    import datasets as ds_module
    original_init = ds_module.Emu3SFTDataset.__init__

    def patched_init(self, args, tokenizer):
        # Temporarily disable raw_image so original_init skips hardcoded path
        original_raw_image = args.raw_image
        args.raw_image = False
        original_init(self, args, tokenizer)
        args.raw_image = original_raw_image

        # Manually set up raw_image with correct vision_hub
        if original_raw_image:
            self.raw_image = True
            self.vision_hub = vision_hub
            self.image_processor = AutoImageProcessor.from_pretrained(
                vision_hub, trust_remote_code=True
            )
            self.image_tokenizer = AutoModel.from_pretrained(
                vision_hub, trust_remote_code=True
            )
            self.image_processor.min_pixels = 80 * 80
            print(f"[patch] vision_hub set → {vision_hub}")

    ds_module.Emu3SFTDataset.__init__ = patched_init


# ── LoRA setup ─────────────────────────────────────────────────────────────────

def apply_lora(model: Emu3MoE, r: int, alpha: int, dropout: float) -> "PeftModel":
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft not installed. Run: pip install peft")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        # Target attention projections in all transformer layers
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Argument dataclasses (HF-style) ───────────────────────────────────────────

@dataclass
class ScriptArgs:
    emu_hub:     str = field(metadata={"help": "UniVLA model path"})
    vision_hub:  str = field(metadata={"help": "Emu3 vision tokenizer path"})
    fast_path:   str = field(metadata={"help": "FAST action tokenizer dir"})
    data_pkl:    str = field(metadata={"help": "Training pickle from step 3"})
    output_dir:  str = field(default="/content/lora_foveated")
    num_epochs:  int = field(default=3)
    batch_size:  int = field(default=1)
    grad_accum:  int = field(default=8)
    lr:          float = field(default=1e-4)
    lora_r:      int = field(default=8)
    lora_alpha:  int = field(default=16)
    lora_dropout: float = field(default=0.05)
    # dataset args (forwarded to DataArguments)
    action_frames: int = field(default=5)
    frames:        int = field(default=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emu-hub",      required=True)
    parser.add_argument("--vision-hub",   required=True)
    parser.add_argument("--fast-path",    required=True)
    parser.add_argument("--data-pkl",     required=True)
    parser.add_argument("--output-dir",   default="/content/lora_foveated")
    parser.add_argument("--num-epochs",   type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--lora-r",       type=int,   default=8)
    parser.add_argument("--lora-alpha",   type=int,   default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Patch dataset vision_hub before importing ──────────────────────────────
    _patch_dataset_vision_hub(args.vision_hub)
    from datasets import Emu3SFTDataset  # noqa: E402  (import after patch)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("[init] Loading tokenizer ...")
    tokenizer = Emu3Tokenizer.from_pretrained(
        args.emu_hub,
        padding_side="right",
        use_fast=False,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("[init] Building dataset ...")

    @dataclass
    class DataArguments:
        data_path:               str   = args.data_pkl
        null_prompt_prob:        float = 0.0
        apply_loss_on_only_vision: bool = False
        apply_loss_on_only_text: bool  = False
        apply_loss_on_only_action: bool = True   # ← only action tokens get loss
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
        raw_image:               bool  = True    # ← on-the-fly VQ tokenization
        post_training:           bool  = False
        datasets_weight:         bool  = False
        without_text:            bool  = False
        real_robot:              bool  = False
        with_cot:                bool  = False

    data_args = DataArguments()
    train_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    print(f"[dataset] {len(train_dataset)} episodes loaded.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[init] Loading Emu3MoE ...")
    model = Emu3MoE.from_pretrained(
        args.emu_hub,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",      # no flash_attn required
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("[lora] Attaching LoRA adapters ...")
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # ── Training arguments ────────────────────────────────────────────────────
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
        dataloader_num_workers=0,  # 0 = main process only (avoids CUDA fork deadlock with raw_image)
        report_to=[],
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = hf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("[train] Starting LoRA fine-tuning ...")
    trainer.train()

    # Save only the LoRA adapter weights (small, ~tens of MB)
    lora_save_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(lora_save_path)
    print(f"\n[done] LoRA adapter saved → {lora_save_path}")
    print("       To use: load base model + apply_lora() + load_adapter(lora_save_path)")


if __name__ == "__main__":
    main()
