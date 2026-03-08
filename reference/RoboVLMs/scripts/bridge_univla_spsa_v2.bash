# SPSA-v2 evaluation script for Bridge tasks
# Uses task-embedding warm-init + cosine-weighted L injection
#
# Usage:
#   conda run --no-capture-output -n univla \
#       bash scripts/bridge_univla_spsa_v2.bash \
#       /path/to/checkpoint \
#       [spsa_threshold] [cosine_temp] [spsa_n]
#
# Examples:
#   Default settings:
#       bash scripts/bridge_univla_spsa_v2.bash /content/pretrain/UniVLA/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K
#
#   Soft weights (cosine_temp=1.0):
#       bash scripts/bridge_univla_spsa_v2.bash /content/pretrain/... 0.40 1.0 20
#
#   Aggressive (lower threshold, more SPSA iters):
#       bash scripts/bridge_univla_spsa_v2.bash /content/pretrain/... 0.50 0.1 30

policy_model=openvla

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOVLMS_ROOT="$(dirname "$SCRIPT_DIR")"
UNIVLA_ROOT=${UNIVLA_ROOT:-$(dirname "$ROBOVLMS_ROOT")/..}
UNIVLA_ROOT="$(cd "$UNIVLA_ROOT" && pwd)"
export PYTHONPATH="${ROBOVLMS_ROOT}:${UNIVLA_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# ---- Args ----
ckpt_dir=$1
vq_hub=${VQ_HUB:-/content/pretrain/Emu3-Stage1}
vision_hub=${VISION_HUB:-/content/pretrain/Emu3-VisionTokenizer}

# SPSA-v2 hyperparams (override via positional args or env vars)
SPSA_THRESHOLD=${2:-${SPSA_THRESHOLD:-0.40}}
COSINE_TEMP=${3:-${COSINE_TEMP:-0.1}}
SPSA_N=${4:-${SPSA_N:-10}}
SPSA_EPSILON=${SPSA_EPSILON:-0.05}
SPSA_ALPHA=${SPSA_ALPHA:-0.01}
SPSA_BETA=${SPSA_BETA:-0.7}
SPSA_MAX_NORM=${SPSA_MAX_NORM:-1.0}
TASK_EMB_SCALE=${TASK_EMB_SCALE:-0.01}
DEBUG_WEIGHT_EVERY=${DEBUG_WEIGHT_EVERY:-0}   # set >0 to print cosine weight stats

SPSA_ARGS="--use-spsa-v2 \
  --spsa-n ${SPSA_N} \
  --spsa-epsilon ${SPSA_EPSILON} \
  --spsa-alpha ${SPSA_ALPHA} \
  --spsa-beta ${SPSA_BETA} \
  --spsa-max-norm ${SPSA_MAX_NORM} \
  --spsa-threshold ${SPSA_THRESHOLD} \
  --task-emb-scale ${TASK_EMB_SCALE} \
  --cosine-temp ${COSINE_TEMP} \
  --debug-weight-every ${DEBUG_WEIGHT_EVERY}"

echo "=== SPSA-v2 config ==="
echo "  threshold:      ${SPSA_THRESHOLD}"
echo "  cosine_temp:    ${COSINE_TEMP}"
echo "  spsa_n:         ${SPSA_N}"
echo "  task_emb_scale: ${TASK_EMB_SCALE}"
echo "======================"

# ---- Task 1: PutCarrot ----
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

python -u eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub ${vq_hub} --vision_hub ${vision_hub} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  ${SPSA_ARGS};

# ---- Task 2: StackCube ----
python -u eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub ${vq_hub} --vision_hub ${vision_hub} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  ${SPSA_ARGS};

# ---- Task 3: PutSpoon ----
python -u eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub ${vq_hub} --vision_hub ${vision_hub} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  ${SPSA_ARGS};

# ---- Task 4: PutEggplant ----
scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

python -u eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub ${vq_hub} --vision_hub ${vision_hub} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  ${SPSA_ARGS};
