#!/usr/bin/env bash

export COMPILE_WARMUP_BATCH_SIZES=""
set -euo pipefail

cd "$(dirname "$0")"

required=(
  "mocap/core/setup_estimator.py"
  "sam_3d_body/__init__.py"
  "sam_3d_body/sam_3d_body_estimator.py"
  "webcam_server.py"
  "webcam.html"
)

for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: missing required file: $file" >&2
    exit 1
  fi
done

python - <<'PY'
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from mocap.core.setup_estimator import build_default_estimator
print("Runtime imports: OK")
PY

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
export FOV_MODEL=s
export FOV_LEVEL=0
export IMG_SIZE=512
export FOV_SIZE=512
export FOV_FAST=1
export FOV_TRT=0
export USE_COMPILE=1
export USE_COMPILE_BACKBONE=1
export DECODER_COMPILE=1
export COMPILE_MODE=reduce-overhead
export MHR_USE_CUDA_GRAPH=0
export GPU_HAND_PREP=1
export SKIP_KEYPOINT_PROMPT=1
export KEYPOINT_PROMPT_INTERM_INTERVAL=999
export BODY_INTERM_PRED_LAYERS=0,1,2
export HAND_INTERM_PRED_LAYERS=0,1
export MHR_NO_CORRECTIVES=1

exec python webcam_server.py
