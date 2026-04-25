#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Foveated Tokenization Experiment — Steps 1~3
#
# 실행 환경 가정:
#   - conda 환경: univla  (Python 3.10.12)
#   - /content/SimplerEnv   : SimplerEnv + ManiSkill2_real2sim + SAPIEN 설치됨
#   - /content/UniVLA       : UniVLA repo
#   - Vulkan / xvfb 설정 완료
#
# 사용법:
#   bash /content/UniVLA/experiments/foveated_tokenization/run.sh
#
# 선택적 인자:
#   TASK        : widowx_put_eggplant_in_basket (default)
#   N_SAMPLES   : 3 (default)
#   OUTPUT_DIR  : /content/foveated_exp (default)
#   DUMMY       : 0/1 — SimplerEnv 없이 더미 이미지로만 테스트 (default: 0)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TASK="${TASK:-widowx_put_eggplant_in_basket}"
N_SAMPLES="${N_SAMPLES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/foveated_exp}"
DUMMY="${DUMMY:-0}"

UNIVLA_DIR="/content/UniVLA"
SCRIPT="${UNIVLA_DIR}/experiments/foveated_tokenization/run_steps123.py"

echo "=================================================="
echo "  Foveated Tokenization Experiment"
echo "  task       : ${TASK}"
echo "  n_samples  : ${N_SAMPLES}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "  dummy mode : ${DUMMY}"
echo "=================================================="

# conda 활성화
source /usr/local/etc/profile.d/conda.sh

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=""
if [ "${DUMMY}" = "1" ]; then
    EXTRA_ARGS="--dummy-image"
    echo "[INFO] Running in dummy mode (no SimplerEnv)"
fi

# MUJOCO_GL + xvfb-run으로 SimplerEnv 렌더링 활성화
export MUJOCO_GL=osmesa

if [ "${DUMMY}" = "1" ]; then
    # dummy 모드는 xvfb 불필요
    conda run -n univla --no-capture-output \
        python "${SCRIPT}" \
            --task    "${TASK}" \
            --n-samples "${N_SAMPLES}" \
            --output-dir "${OUTPUT_DIR}" \
            ${EXTRA_ARGS}
else
    xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
        conda run -n univla --no-capture-output \
            python "${SCRIPT}" \
                --task    "${TASK}" \
                --n-samples "${N_SAMPLES}" \
                --output-dir "${OUTPUT_DIR}" \
                ${EXTRA_ARGS}
fi

echo ""
echo "=================================================="
echo "  Experiment complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "  Files:"
ls "${OUTPUT_DIR}"/*.png 2>/dev/null || echo "  (no PNG files found)"
echo "=================================================="
