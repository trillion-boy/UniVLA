#!/usr/bin/env python3
"""
LatentSaccade 평가 스크립트 — conda run -n univla python latent_saccade_eval.py [args]
sapien 2.x가 conda env에만 있으므로 반드시 conda run으로 실행해야 합니다.
"""
import sys, os, types, argparse, json, time

# ── 경로 등록 ──────────────────────────────────────────────────────────────────
ROOT     = "/content/UniVLA"
EXP      = os.path.join(ROOT, "experiments", "foveated_tokenization")
EMU3     = os.path.join(ROOT, "reference", "Emu3")
SIMPLER  = "/content/SimplerEnv"
MANSKILL = os.path.join(SIMPLER, "ManiSkill2_real2sim")

for p in [ROOT, EXP, EMU3, SIMPLER, MANSKILL]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Patch 1: is_torch_fx_available (new transformers에서 제거됨) ───────────────
import transformers.utils.import_utils as _tui
if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: True

# ── Patch 2: ProcessorMixin 타입 체크 비활성화 ────────────────────────────────
import transformers.processing_utils as _pu
if not getattr(_pu.ProcessorMixin, "_check_patched", False):
    _pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    _pu.ProcessorMixin._check_patched = True

# ── lightning stub ─────────────────────────────────────────────────────────────
if "lightning" not in sys.modules:
    for _n in ["lightning", "lightning.pytorch", "lightning.pytorch.trainer"]:
        sys.modules.setdefault(_n, types.ModuleType(_n))
    class _Trainer:
        pass
    sys.modules["lightning.pytorch.trainer"].Trainer = _Trainer

# ── Patch 3: Emu3Tokenizer.mergeable_ranks ────────────────────────────────────
from emu3.mllm import Emu3Tokenizer
if not hasattr(Emu3Tokenizer, "mergeable_ranks"):
    Emu3Tokenizer.mergeable_ranks = {}

# ── 본 imports ─────────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image as _PIL
import torch
from foveated_inference import LatentSaccadeEmuVLAInference

# ── argparse ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emu-hub",   required=True)
    p.add_argument("--vq-hub",    required=True)
    p.add_argument("--fast-path", required=True)
    p.add_argument("--task",      default="widowx_put_eggplant_in_basket")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--output-dir", default="/content/latent_saccade_eval")
    p.add_argument("--bg-weight",         type=float, default=0.2)
    p.add_argument("--place-src-weight",  type=float, default=0.5)
    p.add_argument("--fovea-weight",      type=float, default=1.0)
    p.add_argument("--min-grasp-steps",   type=int,   default=15)
    p.add_argument("--consec-close",      type=int,   default=3)
    p.add_argument("--min-place-steps",   type=int,   default=8)
    p.add_argument("--dino-cache-steps",  type=int,   default=5)
    p.add_argument("--box-threshold",     type=float, default=0.15)
    p.add_argument("--text-threshold",    type=float, default=0.15)
    p.add_argument("--save-video",        action="store_true")
    p.add_argument("--dino-debug-dir",    default=None)
    p.add_argument("--enable-latent-mask", action="store_true", default=True)
    p.add_argument("--no-overlay",        action="store_true",
                   help="OOD: rgb_overlay 제거 (순수 시뮬 배경)")
    p.add_argument("--overlay-path",      default=None,
                   help="OOD: 다른 overlay 이미지 경로 (절대경로)")
    p.add_argument("--brightness",        type=float, default=1.0,
                   help="OOD: 이미지 밝기 스케일 (1.0=정상, 0.8=어두움 등)")
    return p.parse_args()


TASK_CONFIGS = {
    "widowx_put_eggplant_in_basket": {
        "env_name": "PutEggplantInBasketScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 120,
    },
    "widowx_carrot_on_plate": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
    "widowx_stack_cube": {
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
    "widowx_spoon_on_towel": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5, "sim_freq": 500, "max_episode_steps": 60,
    },
}


def build_env(cfg, ep_id, no_overlay=False, overlay_path=None):
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    robot = cfg["robot"]
    kw = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=cfg["sim_freq"],
        control_mode=get_robot_control_mode(robot, "emu_vla"),
        control_freq=cfg["control_freq"],
        max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
    )
    if not no_overlay:
        if overlay_path and os.path.exists(overlay_path):
            kw["rgb_overlay_path"] = overlay_path
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
        else:
            for base in [SIMPLER, os.path.join(SIMPLER, "ManiSkill2_real2sim")]:
                cand = os.path.join(base, cfg["rgb_overlay_path"])
                if os.path.exists(cand):
                    kw["rgb_overlay_path"] = cand
                    kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
                    break
    env = build_maniskill2_env(cfg["env_name"], **kw)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    return env, obs


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    from PIL import ImageEnhance
    pil = _PIL.fromarray(image)
    return np.array(ImageEnhance.Brightness(pil).enhance(factor))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]

    print(f"[load] LatentSaccadeEmuVLAInference ...", flush=True)
    model = LatentSaccadeEmuVLAInference(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vq_hub,
        device="cuda",
        policy_setup="widowx_bridge",
        fast_path=args.fast_path,
        dino_model="IDEA-Research/grounding-dino-tiny",
        dino_cache_steps=args.dino_cache_steps,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        bbox_margin=2,
        bg_weight=args.bg_weight,
        place_src_weight=args.place_src_weight,
        fovea_weight=args.fovea_weight,
        min_grasp_steps=args.min_grasp_steps,
        consecutive_close_required=args.consec_close,
        min_place_steps=args.min_place_steps,
        enable_latent_mask=args.enable_latent_mask,
        dino_debug_dir=args.dino_debug_dir,
    )
    print(f"[OK] 모델 로드 완료  vis_start={model.vis_start}  vis_end={model.vis_end}", flush=True)

    base_ids = list(range(*task_cfg["obj_episode_range"]))
    ep_ids   = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results  = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────", flush=True)
        env, obs    = build_env(task_cfg, ep_id, no_overlay=args.no_overlay, overlay_path=args.overlay_path)
        instruction = env.get_language_instruction()
        image       = get_image(env, obs, cam_name)
        print(f"   instruction: {instruction}", flush=True)

        model.reset()
        image = apply_brightness(image, args.brightness)
        frames = [image.copy()] if args.save_video else []
        done = truncated = False
        step = 0
        t0   = time.time()

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            raw_actions, env_actions = model.step(image, instruction)
            for raw_a, env_a in zip(raw_actions, env_actions):
                obs, _, done, truncated, _ = env.step(np.concatenate([
                    env_a["world_vector"], env_a["rot_axangle"], env_a["gripper"],
                ]))
                image = apply_brightness(get_image(env, obs, cam_name), args.brightness)
                if args.save_video and step % 4 == 0:
                    frames.append(image.copy())
                new_instr = env.get_language_instruction()
                if new_instr != instruction:
                    instruction = new_instr
                    model.reset()
                step += 1
                if done or truncated:
                    break

        elapsed = time.time() - t0
        status  = "SUCCESS" if done else "FAIL"
        print(f"   → {status}  ({step} steps, {elapsed:.1f}s)", flush=True)
        env.close()

        if args.save_video and frames:
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            pils  = [_PIL.fromarray(f) for f in frames]
            pils[0].save(vpath, save_all=True, append_images=pils[1:], loop=0, duration=100)
            print(f"   GIF: {vpath}", flush=True)

        results.append({
            "ep": ep_count, "ep_id": ep_id,
            "success": bool(done), "steps": step, "elapsed": elapsed,
        })

    n_ok = sum(r["success"] for r in results)
    sr   = n_ok / len(results)
    print(f"\n{'='*50}", flush=True)
    print(f"  task: {args.task}", flush=True)
    print(f"  성공률: {n_ok}/{len(results)} = {sr:.1%}", flush=True)
    print(f"  평균 스텝: {np.mean([r['steps'] for r in results]):.0f}", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        mark = "✓" if r["success"] else "✗"
        print(f"  {mark} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps", flush=True)

    summary = {
        "model": "LatentSaccade",
        "task": args.task,
        "ood_no_overlay": args.no_overlay,
        "ood_overlay_path": args.overlay_path,
        "ood_brightness": args.brightness,
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "config": {
            "fovea_weight": args.fovea_weight,
            "bg_weight": args.bg_weight,
            "place_src_weight": args.place_src_weight,
            "min_grasp_steps": args.min_grasp_steps,
            "consec_close": args.consec_close,
            "dino_cache_steps": args.dino_cache_steps,
            "enable_latent_mask": args.enable_latent_mask,
        },
        "episodes": results,
    }
    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n결과 저장: {save_path}", flush=True)


if __name__ == "__main__":
    main()
