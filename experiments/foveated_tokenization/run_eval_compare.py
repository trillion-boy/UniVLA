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
from typing import Dict, List

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROBOVLMS = os.path.join(_ROOT, "reference", "RoboVLMs")
_EMU3 = os.path.join(_ROOT, "reference", "Emu3")
_SIMPLER = "/content/SimplerEnv"
_MANISKILL = os.path.join(_SIMPLER, "ManiSkill2_real2sim")

for _p in [_ROBOVLMS, _EMU3, _SIMPLER, _MANISKILL]:
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
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "robot_init_xs": [0.147],
        "robot_init_ys": [0.028],
        "robot_init_quats": [[1, 0, 0, 0]],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 3],
        "obs_camera_name": None,
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
    "widowx_carrot_on_plate": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "robot_init_xs": [0.147],
        "robot_init_ys": [0.028],
        "robot_init_quats": [[1, 0, 0, 0]],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 3],
        "obs_camera_name": None,
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
    "widowx_spoon_on_towel": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "robot_init_xs": [0.147],
        "robot_init_ys": [0.028],
        "robot_init_quats": [[1, 0, 0, 0]],
        "obj_variation_mode": "episode",
        "obj_episode_range": [0, 3],
        "obs_camera_name": None,
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
    },
}


def _build_env(task_cfg: dict, robot_x: float, robot_y: float, robot_quat: list, ep_id: int):
    """Build a fresh SimplerEnv episode."""
    from simpler_env.utils.env.env_builder import (
        build_maniskill2_env,
        get_robot_control_mode,
    )

    control_mode = get_robot_control_mode(task_cfg["robot"], "emu_vla")
    env = build_maniskill2_env(
        task_cfg["env_name"],
        obs_mode="rgbd",
        robot=task_cfg["robot"],
        sim_freq=task_cfg["sim_freq"],
        control_mode=control_mode,
        control_freq=task_cfg["control_freq"],
        max_episode_steps=task_cfg["max_episode_steps"],
        scene_name=task_cfg["scene_name"],
    )
    options = {
        "robot_init_options": {
            "init_xy": np.array([robot_x, robot_y]),
            "init_rot_quat": np.array(robot_quat),
        },
        "obj_init_options": {"episode_id": ep_id},
    }
    obs, _ = env.reset(options=options)
    return env, obs, control_mode


def _get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def run_single_episode(
    model,
    task_cfg: dict,
    robot_x: float,
    robot_y: float,
    robot_quat: list,
    ep_id: int,
) -> Dict:
    """Run one episode with `model`. Returns a result dict."""
    cam_name = task_cfg["obs_camera_name"]
    max_steps = task_cfg["max_episode_steps"]

    env, obs, _ = _build_env(task_cfg, robot_x, robot_y, robot_quat, ep_id)
    instruction = env.get_language_instruction()
    image = _get_image(env, obs, cam_name)

    model.reset()
    if hasattr(model, "set_instruction"):
        model.set_instruction(instruction)

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
    return {
        "success": bool(done),
        "steps": step,
        "elapsed": elapsed,
        "ep_id": ep_id,
    }


def evaluate_model(model, task_cfg: dict, n_episodes: int) -> Dict:
    """Evaluate `model` for up to n_episodes and aggregate results."""
    results: List[Dict] = []
    ep_count = 0

    for robot_x in task_cfg["robot_init_xs"]:
        for robot_y in task_cfg["robot_init_ys"]:
            for robot_quat in task_cfg["robot_init_quats"]:
                var_mode = task_cfg["obj_variation_mode"]
                if var_mode == "episode":
                    ep_ids = range(*task_cfg["obj_episode_range"])
                else:
                    ep_ids = range(n_episodes)

                for ep_id in ep_ids:
                    if ep_count >= n_episodes:
                        break
                    r = run_single_episode(
                        model, task_cfg, robot_x, robot_y, robot_quat, ep_id
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
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--blur-scale", type=float, default=0.06)
    # Mode flags
    parser.add_argument(
        "--baseline-only", action="store_true", help="Run only baseline model"
    )
    parser.add_argument(
        "--foveated-only", action="store_true", help="Run only foveated model"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{args.task}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    task_cfg = TASK_CONFIGS[args.task]

    all_results: Dict = {"task": args.task, "args": vars(args), "results": {}}

    # ── Baseline ───────────────────────────────────────────────────────────────
    if not args.foveated_only:
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
        )
        # fast-path override
        if args.fast_path:
            from transformers import AutoProcessor
            pfx = (
                "fast_bridge_t5_s50" if args.policy_setup == "widowx_bridge"
                else "fast_google_a5_s50"
            )
            fp = os.path.join(args.fast_path, pfx)
            if os.path.exists(fp):
                baseline.action_tokenizer = AutoProcessor.from_pretrained(
                    fp, trust_remote_code=True
                )

        baseline_result = evaluate_model(baseline, task_cfg, args.n_episodes)
        all_results["results"]["baseline"] = baseline_result
        print(
            f"\nBaseline success rate: "
            f"{baseline_result['success_rate']:.1%} "
            f"({baseline_result['n_episodes']} eps)"
        )
        del baseline

    # ── Foveated ──────────────────────────────────────────────────────────────
    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("  Foveated EmuVLA (DINO + foveated reconstruction)")
        print("=" * 60)
        from foveated_inference import FoveatedEmuVLAInference

        foveated = FoveatedEmuVLAInference(
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
            blur_scale=args.blur_scale,
        )

        foveated_result = evaluate_model(foveated, task_cfg, args.n_episodes)
        all_results["results"]["foveated"] = foveated_result
        print(
            f"\nFoveated success rate: "
            f"{foveated_result['success_rate']:.1%} "
            f"({foveated_result['n_episodes']} eps)"
        )
        del foveated

    # ── Save + print summary ───────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, r in all_results["results"].items():
        print(
            f"  {name:12s}: {r['success_rate']:.1%}  "
            f"(avg {r['avg_steps']:.0f} steps, {r['avg_time']:.1f}s/ep)"
        )
    if len(all_results["results"]) == 2:
        baseline_sr = all_results["results"]["baseline"]["success_rate"]
        foveated_sr = all_results["results"]["foveated"]["success_rate"]
        delta = foveated_sr - baseline_sr
        print(f"\n  Delta (foveated - baseline): {delta:+.1%}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
