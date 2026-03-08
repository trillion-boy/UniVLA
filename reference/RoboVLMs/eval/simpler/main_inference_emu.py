import os

# Suppress svulkan2/GLFW display errors before any sapien/svulkan2 import
os.environ.setdefault("SVULKAN2_LOG_LEVEL", "off")

import torch
import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from eval.simpler.env_utlis import DictAction
from eval.simpler.maniskill2_evaluator import maniskill2_evaluator
from eval.simpler.model_wrapper import BaseModelInference, EmuVLAInference
from eval.simpler.model_wrapper_emu_spsa_v2 import EmuVLAInferenceSPSA_v2

import argparse
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        type=str,
        default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument(
        "--policy-setup",
        type=str,
        default="google_robot",
        help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--additional-env-save-tags",
        type=str,
        default=None,
        help="Additional tags to save the environment eval results",
    )
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument(
        "--obs-camera-name",
        type=str,
        default=None,
        help="Obtain image observation from this camera for policy input. None = default",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)

    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument(
        "--robot-init-x-range",
        type=float,
        nargs=3,
        default=[0.35, 0.35, 1],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--robot-init-y-range",
        type=float,
        nargs=3,
        default=[0.20, 0.20, 1],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--robot-init-rot-quat-center",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range",
        type=float,
        nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument(
        "--obj-variation-mode",
        type=str,
        default="xy",
        choices=["xy", "episode"],
        help="Whether to vary the xy position of a single object, or to vary predetermined episodes",
    )
    parser.add_argument(
        "--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]"
    )
    parser.add_argument(
        "--obj-init-x-range",
        type=float,
        nargs=3,
        default=[-0.35, -0.12, 5],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--obj-init-y-range",
        type=float,
        nargs=3,
        default=[-0.02, 0.42, 5],
        help="[ymin, ymax, num]",
    )

    parser.add_argument(
        "--additional-env-build-kwargs",
        nargs="+",
        action=DictAction,
        help="Additional env build kwargs in xxx=yyy format. If the value "
        'is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument(
        "--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit"
    )
    parser.add_argument(
        "--octo-init-rng", type=int, default=0, help="Octo init rng seed"
    )

    parser.add_argument(
        "--config_path", type=str, default=None, help="path to the config file"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument("--vq_hub", type=str, default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-Base")
    parser.add_argument("--vision_hub", type=str, default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-VisionVQ")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--double-step", action="store_true")
    parser.add_argument("--CACHE_ROOT", type=str, default="/share/project/yuqi.wang/UniVLA/logs/Simpler")

    # SPSA-v2 flags
    parser.add_argument("--use-spsa-v2", action="store_true",
                        help="Enable SPSA-v2: task-emb warm-init + cosine-weighted injection")
    parser.add_argument("--spsa-n",          type=int,   default=20)
    parser.add_argument("--spsa-epsilon",    type=float, default=0.05)
    parser.add_argument("--spsa-alpha",      type=float, default=0.01)
    parser.add_argument("--spsa-beta",       type=float, default=0.7)
    parser.add_argument("--spsa-max-norm",   type=float, default=1.0)
    parser.add_argument("--spsa-threshold",  type=float, default=0.40)
    parser.add_argument("--task-emb-scale",  type=float, default=0.01)
    parser.add_argument("--cosine-temp",     type=float, default=0.1)
    parser.add_argument("--debug-weight-every", type=int, default=0,
                        help="Print cosine weight stats every N steps (0=off)")

    args = parser.parse_args()

    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append(
                    (
                        Pose(q=euler2quat(r, p, y))
                        * Pose(q=args.robot_init_rot_quat_center)
                    ).q
                )
    # env args: object position
    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    # update logging info (args.additional_env_save_tags) if using a different camera from default
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
        else:
            args.additional_env_save_tags = (
                args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"
            )

    return args


if __name__ == "__main__":
    
    args = get_args()
    CACHE_ROOT = args.CACHE_ROOT
    os.makedirs(CACHE_ROOT, exist_ok=True)
    if 'GOOGLE' in args.emu_hub:
        robot_name = "google"
        policy_setup = "google_robot"
    elif 'BRIDGE' in args.emu_hub:
        robot_name = "bridge"
        policy_setup = "widowx_bridge"
    args.logging_dir = f"results_univla_{robot_name}"
    model_path = args.emu_hub

    args.model_name = 'emu_vla'
    
    if args.double_step:
        args.model_name += "double"
    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    if args.use_spsa_v2:
        args.model_name += "_spsa_v2"
        model = EmuVLAInferenceSPSA_v2(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=torch.device("cuda"),
            policy_setup=policy_setup,
            use_spsa=True,
            spsa_n=args.spsa_n,
            spsa_epsilon=args.spsa_epsilon,
            spsa_alpha=args.spsa_alpha,
            spsa_beta=args.spsa_beta,
            spsa_max_norm=args.spsa_max_norm,
            spsa_threshold=args.spsa_threshold,
            task_emb_scale=args.task_emb_scale,
            cosine_temp=args.cosine_temp,
            debug_weight_every=args.debug_weight_every,
        )
    else:
        model = EmuVLAInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=torch.device("cuda"),
            policy_setup=policy_setup,
        )
    
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
