#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Baseline vs Foveated EmuVLA comparison on SimplerEnv WidowX tasks.
#
# 실행 환경 가정:
#   - conda 환경: univla
#   - /content/SimplerEnv : SimplerEnv + ManiSkill2_real2sim 설치됨
#   - /content/pretrain/  : UniVLA + Emu3-VisionTokenizer 모델 파일들
#   - Grounding DINO deps : install_dino.sh 실행 완료
#
# 사용법:
#   bash /content/UniVLA/experiments/foveated_tokenization/run_eval_compare.sh
#
# 선택적 환경 변수:
#   EMU_HUB        : /content/pretrain/UniVLA
#   VQ_HUB         : /content/pretrain/Emu3-VisionTokenizer
#   VISION_HUB     : /content/pretrain/Emu3-VisionTokenizer
#   FAST_PATH      : /content/pretrain
#   TASK           : widowx_put_eggplant_in_basket
#   N_EPISODES     : 10
#   OUTPUT_DIR     : /content/foveated_eval
#   BASELINE_ONLY  : 0/1
#   FOVEATED_ONLY  : 0/1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EMU_HUB="${EMU_HUB:-/content/pretrain/UniVLA}"
VQ_HUB="${VQ_HUB:-/content/pretrain/Emu3-VisionTokenizer}"
VISION_HUB="${VISION_HUB:-/content/pretrain/Emu3-VisionTokenizer}"
FAST_PATH="${FAST_PATH:-/content/pretrain}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
N_EPISODES="${N_EPISODES:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/foveated_eval}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"
FOVEATED_ONLY="${FOVEATED_ONLY:-0}"

UNIVLA_DIR="/content/UniVLA"
SCRIPT="${UNIVLA_DIR}/experiments/foveated_tokenization/run_eval_compare.py"

echo "=================================================="
echo "  Baseline vs Foveated EmuVLA Comparison"
echo "  task        : ${TASK}"
echo "  n_episodes  : ${N_EPISODES}"
echo "  output_dir  : ${OUTPUT_DIR}"
echo "  emu_hub     : ${EMU_HUB}"
echo "  fast_path   : ${FAST_PATH}"
echo "=================================================="

source /usr/local/etc/profile.d/conda.sh
mkdir -p "${OUTPUT_DIR}"

# Build optional flags
EXTRA_ARGS=""
if [ "${BASELINE_ONLY}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --baseline-only"
fi
if [ "${FOVEATED_ONLY}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --foveated-only"
fi

export MUJOCO_GL=osmesa

xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
    conda run -n univla --no-capture-output \
        python "${SCRIPT}" \
            --emu-hub    "${EMU_HUB}" \
            --vq-hub     "${VQ_HUB}" \
            --vision-hub "${VISION_HUB}" \
            --fast-path  "${FAST_PATH}" \
            --task       "${TASK}" \
            --n-episodes "${N_EPISODES}" \
            --output-dir "${OUTPUT_DIR}" \
            ${EXTRA_ARGS}

echo ""
echo "=================================================="
echo "  Experiment complete!"
echo "  Results: ${OUTPUT_DIR}/results_${TASK}.json"
echo "=================================================="
