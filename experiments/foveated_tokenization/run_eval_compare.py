"""
Baseline vs Foveated inference comparison on SimplerEnv WidowX tasks.

Runs both models for N episodes and reports per-model success rates.
Results are saved to JSON in output_dir.

Usage (Colab):
    python experiments/foveated_tokenization/run_eval_compare.py \
        --emu-hub   /content/pretrain/UniVLA \
        --vq-hub    /content/pretrain/Emu3-VisionTokenizer \
        --vision-hub /content/pretrain/Emu3-VisionTokenizer \
        --fast-path /content/pretrain \
        --task      widowx_put_eggplant_in_basket \
        --n-episodes 10 \
        --output-dir /content/foveated_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROBOVLMS = os.path.join(_ROOT, "reference", "RoboVLMs")
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
_SIMPLER = "/content/SimplerEnv"
_MANISKILL = os.path.join(_SIMPLER, "ManiSkill2_real2sim")

for _p in [_ROOT, _ROBOVLMS, _EMU3, _SIMPLER, _MANISKILL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── lightning stub (calvin model_wrapper imports it at module level) ────────────
def _ensure_lightning_stub() -> None:
    import types
    if "lightning" not in sys.modules:
        for name in ["lightning", "lightning.pytorch", "lightning.pytorch.trainer"]:
            sys.modules.setdefault(name, types.ModuleType(name))
        class _Trainer:
            pass
        sys.modules["lightning.pytorch.trainer"].Trainer = _Trainer

_ensure_lightning_stub()

# ── Task configurations ────────────────────────────────────────────────────────
# Maps task shorthand → SimplerEnv build parameters.
TASK_CONFIGS: Dict[str, dict] = {
    "widowx_put_eggplant_in_basket": {
        "env_name": "PutEggplantInBasketScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
    "widowx_carrot_on_plate": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
    "widowx_spoon_on_towel": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
    "widowx_stack_cube": {
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 60,
    },
}


def _build_env(task_cfg: dict, ep_id: int):
    """Build a fresh SimplerEnv episode."""
    import os
    from simpler_env.utils.env.env_builder import (
        build_maniskill2_env,
        get_robot_control_mode,
    )

    robot = task_cfg["robot"]
    control_mode = get_robot_control_mode(robot, "emu_vla")
    build_kwargs = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=task_cfg["sim_freq"],
        control_mode=control_mode,
        control_freq=task_cfg["control_freq"],
        max_episode_steps=task_cfg["max_episode_steps"],
    )
    if task_cfg.get("scene_name"):
        build_kwargs["scene_name"] = task_cfg["scene_name"]

    # Visual matching: overlay real background image so sim looks like real robot
    overlay_rel = task_cfg.get("rgb_overlay_path")
    if overlay_rel:
        # Resolve relative to SimplerEnv ManiSkill2 root
        for base in ["/content/SimplerEnv", "/content/SimplerEnv/ManiSkill2_real2sim/.."]:
            candidate = os.path.join(base, overlay_rel)
            if os.path.exists(candidate):
                build_kwargs["rgb_overlay_path"] = candidate
                build_kwargs["rgb_overlay_cameras"] = task_cfg.get(
                    "rgb_overlay_cameras", ["3rd_view_camera"]
                )
                # rgb_overlay needs segmentation info to mask objects
                build_kwargs["camera_cfgs"] = {"add_segmentation": True}
                print(f"[env] rgb_overlay: {candidate}")
                break
        else:
            print(f"[env] WARNING: rgb_overlay not found at {overlay_rel}, skipping")

    env = build_maniskill2_env(task_cfg["env_name"], **build_kwargs)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    return env, obs, control_mode


def _get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def run_single_episode(
    model,
    task_cfg: dict,
    ep_id: int,
    video_path: Optional[str] = None,
) -> Dict:
    """Run one episode with `model`. Returns a result dict."""
    cam_name = task_cfg["obs_camera_name"]
    max_steps = task_cfg["max_episode_steps"]

    env, obs, _ = _build_env(task_cfg, ep_id)
    instruction = env.get_language_instruction()
    image = _get_image(env, obs, cam_name)

    model.reset()
    if hasattr(model, "set_instruction"):
        model.set_instruction(instruction)

    frames: List[np.ndarray] = []
    if video_path:
        frames.append(image.copy())

    done = False
    truncated = False
    step = 0
    t0 = time.time()

    while not (done or truncated) and step < max_steps:
        raw_actions, env_actions = model.step(image, instruction)
        for raw_action, env_action in zip(raw_actions, env_actions):
            obs, _reward, done, truncated, info = env.step(
                np.concatenate([
                    env_action["world_vector"],
                    env_action["rot_axangle"],
                    env_action["gripper"],
                ])
            )
            image = _get_image(env, obs, cam_name)
            if video_path and step % 4 == 0:  # save every 4 steps to keep GIF small
                frames.append(image.copy())

            # Long-horizon task: check subtask transitions
            is_final = env.is_final_subtask()
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                model.reset()
                if hasattr(model, "set_instruction"):
                    model.set_instruction(instruction)

            step += 1
            if done or truncated:
                break

    elapsed = time.time() - t0
    env.close()

    if video_path and frames:
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                video_path, save_all=True, append_images=pil_frames[1:],
                loop=0, duration=100,
            )
            print(f"    Video saved: {video_path}")
        except Exception as e:
            print(f"    Video save failed: {e}")

    return {
        "success": bool(done),
        "steps": step,
        "elapsed": elapsed,
        "ep_id": ep_id,
    }


def evaluate_model(
    model, task_cfg: dict, n_episodes: int,
    model_name: str = "model", video_dir: Optional[str] = None,
) -> Dict:
    """Evaluate `model` for up to n_episodes and aggregate results."""
    results: List[Dict] = []
    ep_count = 0

    var_mode = task_cfg["obj_variation_mode"]
    if var_mode == "episode":
        base_ids = list(range(*task_cfg["obj_episode_range"]))
        # Cycle through base IDs to fill n_episodes
        ep_ids = [base_ids[i % len(base_ids)] for i in range(n_episodes)]
    else:
        ep_ids = list(range(n_episodes))

    for ep_id in ep_ids:
        if ep_count >= n_episodes:
            break
        vpath = None
        if video_dir:
            vpath = os.path.join(video_dir, f"{model_name}_ep{ep_count:02d}.gif")
        r = run_single_episode(
            model, task_cfg, ep_id,
            video_path=vpath,
        )
        results.append(r)
        status = "SUCCESS" if r["success"] else "FAIL"
        print(
            f"    ep {ep_count} (id={ep_id}): {status} "
            f"({r['steps']} steps, {r['elapsed']:.1f}s)"
        )
        ep_count += 1

    successes = [r["success"] for r in results]
    return {
        "episodes": results,
        "n_episodes": len(results),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "avg_time": float(np.mean([r["elapsed"] for r in results])) if results else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs Foveated EmuVLA comparison on SimplerEnv"
    )
    parser.add_argument("--emu-hub", required=True, help="UniVLA model path")
    parser.add_argument("--vq-hub", required=True, help="VQ tokenizer path")
    parser.add_argument("--vision-hub", required=True, help="Vision encoder path")
    parser.add_argument(
        "--fast-path", default=None,
        help="Base dir containing fast_bridge_t5_s50 / fast_google_a5_s50"
    )
    parser.add_argument(
        "--task", default="widowx_put_eggplant_in_basket",
        choices=list(TASK_CONFIGS.keys()),
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="/content/foveated_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy-setup", default="widowx_bridge",
        choices=["widowx_bridge", "google_robot"],
    )
    # Foveated-specific
    parser.add_argument(
        "--dino-model", default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for Grounding DINO"
    )
    parser.add_argument("--dino-cache-steps", type=int, default=5)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--blur-scale", type=float, default=0.06)
    parser.add_argument(
        "--fovea-fraction", type=float, default=0.4,
        help="Token-fovea radius as fraction of token grid (0.4 = center 50%% of image area)"
    )
    # Mode flags
    parser.add_argument(
        "--baseline-only", action="store_true", help="Run only baseline model"
    )
    parser.add_argument(
        "--foveated-only", action="store_true", help="Run only image-level foveated model"
    )
    parser.add_argument(
        "--token-fovea-only", action="store_true",
        help="Run only token-level foveated model (no image blurring)"
    )
    parser.add_argument(
        "--true-fovea-only", action="store_true",
        help="Run only true foveated model (center crop upscale + peripheral context)"
    )
    parser.add_argument(
        "--dual-fovea-only", action="store_true",
        help="Run only dual-object true foveated model (crops around midpoint of src+dst objects)"
    )
    parser.add_argument(
        "--bass-only", action="store_true",
        help="Run only BASS Möbius-warp model (single warped image, no token mixing)"
    )
    parser.add_argument("--grasp-strength", type=float, default=4.0,
        help="BASS magnification during grasping phase (default 4.0)")
    parser.add_argument("--move-strength", type=float, default=2.0,
        help="BASS magnification during moving phase (default 2.0)")
    parser.add_argument(
        "--crop-fraction", type=float, default=0.5,
        help="Center crop size as fraction of image dimension (default 0.5 = 2x resolution)"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save per-episode GIF videos to output-dir"
    )
    parser.add_argument(
        "--dino-debug-dir", default=None,
        help="Directory to save DINO detection overlay images for debugging"
    )
    parser.add_argument(
        "--lora-path", default=None,
        help="Path to LoRA adapter directory (output of train_lora_foveated.py). "
             "Applied to the foveated model only."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{args.task}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    task_cfg = TASK_CONFIGS[args.task]

    all_results: Dict = {"task": args.task, "args": vars(args), "results": {}}

    # Determine which models to run: if any *_only flag is set, run only that one.
    any_only = any([
        args.baseline_only, args.foveated_only, args.token_fovea_only,
        args.true_fovea_only, args.dual_fovea_only, args.bass_only,
    ])

    # ── Baseline ───────────────────────────────────────────────────────────────
    if args.baseline_only or not any_only:
        print("\n" + "=" * 60)
        print("  Baseline EmuVLA")
        print("=" * 60)
        # Import from foveated_inference (standalone, no robovlms/lightning deps)
        _exp_dir = os.path.join(_ROOT, "experiments", "foveated_tokenization")
        if _exp_dir not in sys.path:
            sys.path.insert(0, _exp_dir)
        from foveated_inference import EmuVLAInference

        baseline = EmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
        )

        baseline_result = evaluate_model(
            baseline, task_cfg, args.n_episodes,
            model_name="baseline",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["baseline"] = baseline_result
        print(
            f"\nBaseline success rate: "
            f"{baseline_result['success_rate']:.1%} "
            f"({baseline_result['n_episodes']} eps)"
        )
        del baseline
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Image-level Foveated ──────────────────────────────────────────────────
    if args.foveated_only or not any_only:
        print("\n" + "=" * 60)
        print("  Foveated EmuVLA (image-level: DINO + blur)")
        print("=" * 60)
        from foveated_inference import FoveatedEmuVLAInference

        foveated = FoveatedEmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
            lora_path=args.lora_path,
            dino_model=args.dino_model,
            dino_cache_steps=args.dino_cache_steps,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            blur_scale=args.blur_scale,
            dino_debug_dir=args.dino_debug_dir,
        )

        foveated_result = evaluate_model(
            foveated, task_cfg, args.n_episodes,
            model_name="foveated",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["foveated"] = foveated_result
        print(
            f"\nFoveated (image-level) success rate: "
            f"{foveated_result['success_rate']:.1%} "
            f"({foveated_result['n_episodes']} eps)"
        )
        del foveated
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Token-level Foveated ──────────────────────────────────────────────────
    if args.token_fovea_only or not any_only:
        print("\n" + "=" * 60)
        print(f"  Token-Foveated EmuVLA (token-level, fovea_fraction={args.fovea_fraction})")
        print("=" * 60)
        from foveated_inference import TokenFoveatedEmuVLAInference

        token_foveated = TokenFoveatedEmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
            dino_model=args.dino_model,
            dino_cache_steps=args.dino_cache_steps,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            fovea_fraction=args.fovea_fraction,
            dino_debug_dir=args.dino_debug_dir,
        )

        token_fovea_result = evaluate_model(
            token_foveated, task_cfg, args.n_episodes,
            model_name="token_fovea",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["token_fovea"] = token_fovea_result
        print(
            f"\nToken-Foveated success rate: "
            f"{token_fovea_result['success_rate']:.1%} "
            f"({token_fovea_result['n_episodes']} eps)"
        )
        del token_foveated
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── True Foveated ─────────────────────────────────────────────────────────
    if args.true_fovea_only or not any_only:
        print("\n" + "=" * 60)
        print(f"  True-Foveated EmuVLA (crop={args.crop_fraction}, fovea={args.fovea_fraction})")
        print("=" * 60)
        from foveated_inference import TrueFoveatedEmuVLAInference

        true_foveated = TrueFoveatedEmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
            dino_model=args.dino_model,
            dino_cache_steps=args.dino_cache_steps,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            crop_fraction=args.crop_fraction,
            fovea_fraction=args.fovea_fraction,
            dino_debug_dir=args.dino_debug_dir,
        )

        true_fovea_result = evaluate_model(
            true_foveated, task_cfg, args.n_episodes,
            model_name="true_fovea",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["true_fovea"] = true_fovea_result
        print(
            f"\nTrue-Foveated success rate: "
            f"{true_fovea_result['success_rate']:.1%} "
            f"({true_fovea_result['n_episodes']} eps)"
        )
        del true_foveated

    # ── Dual-Object True Foveated ─────────────────────────────────────────────
    if args.dual_fovea_only or not any_only:
        print("\n" + "=" * 60)
        print(f"  Dual-Obj True-Foveated EmuVLA (crop={args.crop_fraction}, fovea={args.fovea_fraction})")
        print("=" * 60)
        from foveated_inference import TrueFoveatedDualObjEmuVLAInference

        dual_foveated = TrueFoveatedDualObjEmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
            dino_model=args.dino_model,
            dino_cache_steps=args.dino_cache_steps,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            crop_fraction=args.crop_fraction,
            fovea_fraction=args.fovea_fraction,
            dino_debug_dir=args.dino_debug_dir,
        )

        dual_fovea_result = evaluate_model(
            dual_foveated, task_cfg, args.n_episodes,
            model_name="dual_fovea",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["dual_fovea"] = dual_fovea_result
        print(
            f"\nDual-Obj Foveated success rate: "
            f"{dual_fovea_result['success_rate']:.1%} "
            f"({dual_fovea_result['n_episodes']} eps)"
        )
        del dual_foveated
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── BASS (Möbius warp) ────────────────────────────────────────────────────
    if args.bass_only or not any_only:
        print("\n" + "=" * 60)
        print(f"  BASS EmuVLA (Möbius warp, grasp_s={args.grasp_strength}, move_s={args.move_strength})")
        print("=" * 60)
        from foveated_inference import BASSEmuVLAInference

        bass_model = BASSEmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=args.device,
            policy_setup=args.policy_setup,
            fast_path=args.fast_path,
            dino_model=args.dino_model,
            dino_cache_steps=args.dino_cache_steps,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            grasp_strength=args.grasp_strength,
            move_strength=args.move_strength,
            dual_focus=True,
            dino_debug_dir=args.dino_debug_dir,
        )

        bass_result = evaluate_model(
            bass_model, task_cfg, args.n_episodes,
            model_name="bass",
            video_dir=args.output_dir if args.save_video else None,
        )
        all_results["results"]["bass"] = bass_result
        print(
            f"\nBASS success rate: "
            f"{bass_result['success_rate']:.1%} "
            f"({bass_result['n_episodes']} eps)"
        )
        del bass_model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save + print summary ───────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, r in all_results["results"].items():
        print(
            f"  {name:14s}: {r['success_rate']:.1%}  "
            f"(avg {r['avg_steps']:.0f} steps, {r['avg_time']:.1f}s/ep)"
        )
    baseline_sr = all_results["results"].get("baseline", {}).get("success_rate")
    if baseline_sr is not None:
        for name in ("foveated", "token_fovea", "true_fovea", "dual_fovea", "bass"):
            if name in all_results["results"]:
                delta = all_results["results"][name]["success_rate"] - baseline_sr
                print(f"\n  Delta ({name} - baseline): {delta:+.1%}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
