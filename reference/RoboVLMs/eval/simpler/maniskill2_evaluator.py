"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler
from PIL import Image as PILImage, ImageDraw, ImageFont

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    model_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=True,
    additional_env_save_tags=None,
    logging_dir="./results_V2",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    # print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset()

    timestep = 0
    success = "failure"

    confidences = []
    spsa_fired_count = 0
    conf_log_every = 10  # print per-step conf every N steps

    def _overlay_confidence(img, conf):
        pil_img = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        text = f"conf: {conf:.3f}"
        draw.rectangle([2, 2, 120, 16], fill=(0, 0, 0))
        draw.text((4, 3), text, fill=(255, 255, 0))
        return np.array(pil_img)

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, task_description)
        step_confidence = getattr(model, 'last_confidence', 0.0)
        rolling_conf = getattr(model, 'rolling_conf', None)
        spsa_threshold = getattr(model, 'spsa_threshold', None)
        use_spsa = getattr(model, 'use_spsa', False)
        spsa_fired = use_spsa and rolling_conf is not None and spsa_threshold is not None and rolling_conf < spsa_threshold
        if spsa_fired:
            spsa_fired_count += 1
        if timestep % conf_log_every == 0:
            rolling_str = f"  rolling={rolling_conf:.3f}" if rolling_conf is not None else ""
            spsa_str = f"  SPSA={'ON' if spsa_fired else 'off'}" if use_spsa else ""
            print(f"  [step {timestep:3d}] conf={step_confidence:.3f}{rolling_str}{spsa_str}")

        # action chunk
        raw_action_list = raw_action
        action_list = action

        for raw_action, action in zip(raw_action_list, action_list):
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)

            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            # step the environment
            obs, reward, done, truncated, info = env.step(
                np.concatenate(
                    [action["world_vector"], action["rot_axangle"], action["gripper"]]
                ),
            )

            success = "success" if done else "failure"
            if done:
                predicted_terminated = True
                break
            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                print(task_description)
                model.reset()

            is_final_subtask = env.is_final_subtask()

            if not is_final_subtask and info["episode_stats"].get("is_drawer_open", False):
                env.advance_to_next_subtask()

            image = get_image_from_maniskill2_obs_dict(
                env, obs, camera_name=obs_camera_name
            )
            confidences.append(step_confidence)
            images.append(_overlay_confidence(image, step_confidence))
            timestep += 1
    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"

    ckpt_path_basename = f"{model_name}"

    _stat_abbrev = {
        "moved_correct_obj": "mc",
        "moved_wrong_obj": "mw",
        "is_src_obj_grasped": "gr",
        "consecutive_grasp": "cg",
        "src_on_target": "ot",
    }

    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_ep{obj_episode_id}"
    for k, v in episode_stats.items():
        key = _stat_abbrev.get(k, k)
        val = "T" if v is True else ("F" if v is False else v)
        video_name = video_name + f"_{key}-{val}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save GIF
    gif_path = video_path.replace(".mp4", ".gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    pil_frames = [PILImage.fromarray(img) for img in images]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=200,
        loop=0,
        optimize=False,
    )
    confs = np.array(confidences)
    p25, p75 = np.percentile(confs, 25), np.percentile(confs, 75)
    spsa_info = f"  spsa_fired={spsa_fired_count}/{len(confs)}steps" if getattr(model, 'use_spsa', False) else ""
    print(f"[GIF] mean={np.mean(confs):.3f}  min={np.min(confs):.3f}  max={np.max(confs):.3f}  p25={p25:.3f}  p75={p75:.3f}{spsa_info}  → {gif_path}")
    # per-step conf timeline (every 10 steps)
    if len(confs) > 0:
        timeline = "  ".join(f"s{i*conf_log_every}:{confs[i*conf_log_every]:.3f}" for i in range(len(confs) // conf_log_every + 1) if i * conf_log_every < len(confs))
        print(f"[CONF timeline] {timeline}")

    # save action trajectory
    # action_path = video_path.replace(".mp4", ".png")
    # action_root = os.path.dirname(action_path) + "/actions/"
    # os.makedirs(action_root, exist_ok=True)
    # action_path = action_root + os.path.basename(action_path)
    # model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    model_name = args.model_name
    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    model_name=model_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr
