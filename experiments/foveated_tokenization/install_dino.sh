#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Install Grounding DINO dependencies into the univla conda environment.
#
# Grounding DINO is accessed via the HuggingFace Transformers
# AutoModelForZeroShotObjectDetection API (available since transformers 4.38).
#
# Usage:
#   bash /content/UniVLA/experiments/foveated_tokenization/install_dino.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /usr/local/etc/profile.d/conda.sh

echo "=== Upgrading transformers (need >= 4.38 for Grounding DINO) ==="
conda run -n univla --no-capture-output \
    pip install "transformers>=4.38.0" --upgrade

echo "=== Installing torchvision (needed by DINO image processor) ==="
conda run -n univla --no-capture-output \
    pip install torchvision --quiet

echo "=== Verifying installation ==="
conda run -n univla --no-capture-output python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
print('transformers DINO import: OK')
import torchvision
print(f'torchvision: {torchvision.__version__}')
"

echo ""
echo "=== Done! ==="
echo "Quick smoke test:"
echo "  conda run -n univla python -c \""
echo "    from experiments.foveated_tokenization.grounding_dino_wrapper import GroundingDINOWrapper"
echo "    import numpy as np"
echo "    dino = GroundingDINOWrapper()"
echo "    img = np.zeros((256,256,3), dtype='uint8')"
echo "    cx, cy = dino.get_fovea_center(img, 'pick up eggplant')"
echo "    print(f'fovea center: ({cx},{cy})')"
echo "  \""
