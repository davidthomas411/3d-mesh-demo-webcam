@echo off
rem Local real-time launcher (Windows, single NVIDIA GPU). Tested on an RTX 4060 8 GB.
rem Needs the Fast-SAM-3D-Body checkout next to this repo, with its .venv, checkpoints and the
rem 384 TensorRT backbone engine (python build_backbone_trt_local.py 384).
setlocal
set "DEMO=%~dp0"
set "FAST=%~dp0..\Fast-SAM-3D-Body"
cd /d "%FAST%"
set PYTHONPATH=%FAST%
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
if not defined HOST set HOST=127.0.0.1
if not defined PORT set PORT=8097
rem model: 384 input, TensorRT FP16 backbone, all three intermediate body predictions (accuracy)
set IMG_SIZE=384
set USE_TRT_BACKBONE=1
set TRT_BACKBONE_PATH=./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16_384.engine
set USE_COMPILE=0
set BODY_INTERM_PRED_LAYERS=0,1,2
set HAND_INTERM_PRED_LAYERS=0,1
set KEYPOINT_PROMPT_INTERM_INTERVAL=999
set SKIP_KEYPOINT_PROMPT=1
set GPU_HAND_PREP=1
set MHR_NO_CORRECTIVES=1
set MHR_USE_CUDA_GRAPH=0
set FOV_MODEL=s
set FOV_LEVEL=0
set FOV_SIZE=512
set FOV_FAST=1
set FOV_TRT=0
rem server: person box from the previous mesh, detector only every 30 frames, no per-frame debug prints
set TRACK_BOX=1
set DETECT_EVERY=30
set QUIET=1
"%FAST%\.venv\Scripts\python.exe" "%DEMO%webcam_server.py" %*
