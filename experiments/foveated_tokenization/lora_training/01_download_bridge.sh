#!/bin/bash
# Download bridge_orig dataset from Open X-Embodiment (Google Cloud Storage).
# Requires: gcloud CLI authenticated, or use gsutil with public access.
#
# Total size: ~60 GB → save to /local-scratch (368 GB available)
#
# Usage:
#   bash 01_download_bridge.sh
#   bash 01_download_bridge.sh --output /local-scratch/bridge_orig

set -e

OUTPUT_DIR="${1:-/local-scratch/bridge_orig}"
GCS_PATH="gs://gresearch/robotics/bridge_orig/0.1.0"

echo "=== Bridge dataset download ==="
echo "Source : $GCS_PATH"
echo "Dest   : $OUTPUT_DIR"
echo "Est. size: ~60 GB"
echo ""

mkdir -p "$OUTPUT_DIR"

# Install gsutil if not present
if ! command -v gsutil &>/dev/null; then
    echo "[install] gsutil not found, installing google-cloud-storage ..."
    pip install -q google-cloud-storage
    pip install -q gsutil
fi

echo "[download] Starting ... (this takes 1-2 hours depending on bandwidth)"
gsutil -m cp -r "$GCS_PATH" "$OUTPUT_DIR"

echo ""
echo "[done] Bridge dataset saved to: $OUTPUT_DIR"
echo "       Next step: python 02_process_and_foveate.py --dataset-dir $OUTPUT_DIR/0.1.0 --output-dir /local-scratch/bridge_foveated"
