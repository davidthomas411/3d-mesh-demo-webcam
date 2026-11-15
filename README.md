# Fast SAM 3D Body Webcam Demo

Live browser webcam inference using Fast SAM 3D Body on a remote NVIDIA GPU server.

![screenshot](docs/demo-screenshot.png)

## Overview

```text
Browser webcam
  -> JPEG frame over WebSocket
  -> Coder forwarded port
  -> persistent Fast SAM 3D Body estimator
  -> binary float32 mesh vertices
  -> projected overlay and interactive 3D rendering
```

The interface provides:

- Live webcam video
- Camera-aligned mesh overlay
- Independent interactive 3D mesh
- Runtime inference controls
- Current, median, and P90 latency measurements

## Upstream repository

This demo runs inside a working checkout of:

[Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body)

Project page:

[Fast SAM 3D Body project page](https://yangtiming.github.io/Fast-SAM-3D-Body-Page/)

## Requirements

The tested environment uses:

- Python 3.11
- PyTorch 2.5.1 with CUDA 12.4
- Detectron2 0.6
- MoGe2
- Ultralytics
- Flask
- flask-sock
- OpenCV
- NumPy
- NVIDIA RTX A6000

The Fast SAM 3D Body checkpoint must exist at:

```text
checkpoints/sam-3d-body-dinov3/model.ckpt
checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

The default pose detector is:

```text
checkpoints/yolo/yolo11m-pose.pt
```

The model repository is gated on Hugging Face. Access approval and Hugging Face authentication are required before downloading the checkpoint.

## Installation

Clone the upstream repository:

```bash
git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git
cd Fast-SAM-3D-Body
```

Activate the Conda environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fast_sam_3d_body
```

Install the web dependencies into the active environment:

```bash
python -m pip install Flask flask-sock
```

Copy these files into the Fast SAM 3D Body repository root:

```text
webcam_server.py
webcam.html
```

## Model files

Authenticate to Hugging Face:

```bash
hf auth login
```

The required local directory should contain:

```text
checkpoints/
└── sam-3d-body-dinov3/
    ├── model.ckpt
    ├── model_config.yaml
    └── assets/
        └── mhr_model.pt
```

Download the YOLO11m-Pose detector if necessary:

```bash
python -c "from ultralytics import YOLO; YOLO('checkpoints/yolo/yolo11m-pose.pt')"
```

## EGL requirement

Headless rendering imports require the generic EGL loader.

On Ubuntu 24.04:

```bash
sudo apt-get install --reinstall libegl1
```

Verify EGL:

```bash
ldconfig -p 2>/dev/null | grep -E 'libEGL(\.so|_nvidia)'
```

Expected libraries include:

```text
libEGL.so.1
libEGL_nvidia.so.0
```

## Start the server

Activate the environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fast_sam_3d_body
```

Start the persistent inference server:

```bash
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=0 \
TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas" \
FOV_MODEL=s \
FOV_LEVEL=0 \
FOV_SIZE=512 \
FOV_FAST=1 \
FOV_TRT=0 \
USE_COMPILE=1 \
USE_COMPILE_BACKBONE=1 \
DECODER_COMPILE=1 \
COMPILE_MODE=reduce-overhead \
COMPILE_WARMUP_BATCH_SIZES=1 \
MHR_USE_CUDA_GRAPH=0 \
GPU_HAND_PREP=1 \
SKIP_KEYPOINT_PROMPT=1 \
KEYPOINT_PROMPT_INTERM_INTERVAL=999 \
BODY_INTERM_PRED_LAYERS=0,1,2 \
HAND_INTERM_PRED_LAYERS=0,1 \
MHR_NO_CORRECTIVES=1 \
python webcam_server.py
```

Wait for:

```text
ESTIMATOR_READY vertices=18439 faces=36874
```

The first use of a new compiled inference path may be much slower than steady-state execution.

## Triton and CUDA

The tested host exposed CUDA 13.2, while the Conda environment contained CUDA 12.4 and Triton 3.1.0.

The following setting forces Triton to use the compatible Conda CUDA assembler:

```bash
TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
```

Verify the selected assembler:

```bash
TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas" \
python -c "from triton.backends.nvidia.compiler import _path_to_binary; print(_path_to_binary('ptxas'))"
```

Expected result:

```text
(<conda-environment>/bin/ptxas, 12.4)
```

## Coder port forwarding

The server listens on:

```text
0.0.0.0:8097
```

Configure Coder to forward server port `8097`.

The tested local browser address was:

```text
http://localhost:8098/
```

Open the page, then select **Start webcam and inference**.

## Runtime controls

### Inference mode

- **Body only:** lowest latency
- **Full body + hands:** processes body and hand crops

The first use of each mode may trigger additional `torch.compile` work.

### Camera intrinsics

- **Fixed calibrated median:** calibrates focal length and reuses the median
- **Dynamic MoGe per frame:** estimates focal length for every frame

The tested calibration window is 30 valid frames.

### Independent 3D view

- **Lock 3D center:** frames the mesh once and avoids visible center jitter
- **Reset 3D camera:** recalculates the initial framing
- Mouse drag rotates the mesh
- Mouse wheel zooms the mesh

Locking the independent view does not smooth or modify the predicted vertices.

### Overlay controls

- Show or hide the projected overlay
- Select dense, medium, or sparse topology
- Adjust opacity
- Adjust line width

The overlay uses predicted camera translation and focal length to project the mesh into webcam coordinates.

## Latency HUD

The HUD displays non-overlapping latency components:

```text
JPEG encode
Server queue
JPEG decode
Inference
Network + HTTP
Browser render
TOTAL
```

Each component reports:

- Current value
- Rolling median
- Rolling P90

All values are milliseconds.

## Observed performance

Observed on an NVIDIA RTX A6000 after compilation and focal calibration:

```text
Body-only inference at 448 or 512 input: approximately 75-100 ms
End-to-end latency: approximately 140-170 ms
```

Full body plus hands was slower in the tested configuration.

Results vary with:

- GPU contention
- Subject count
- Inference mode
- Model input size
- Compilation state
- Browser workload
- Overlay density

These values are deployment observations, not guaranteed benchmarks.

## Architecture notes

- The estimator is loaded once and reused.
- Frames and meshes use a persistent WebSocket connection.
- JPEG frames are transferred as binary data.
- Mesh vertices are returned as little-endian float32 values.
- Mesh topology is transferred once.
- The browser updates an existing Three.js geometry buffer.
- The server does not reload the model for each frame.
- OpenMP and MKL are limited to eight threads to prevent CPU oversubscription.
- Fixed median intrinsics remove repeated MoGe execution.
- Body-only mode avoids the two additional hand crops used by full mode.
- The independent mesh can use a fixed center without temporal smoothing.

## Femto Bolt depth fusion (experimental)

The RGB-only mesh has an ambiguous overall scale/depth (`pred_cam_t`). An Orbbec Femto Bolt
supplies metric depth, so the server can fix that ambiguity and refine the pose of the mesh.

```text
Femto Bolt RGB + depth (same device: factory extrinsics, hardware sync)
  -> SAM 3D Body on the Bolt's RGB            (pose + shape, slow path)
  -> depth anchor: one robust scale about the camera origin   (< 1 ms)
  -> projective point-to-plane ICP, 2-4 iterations            (a few ms, numpy)
  -> metric mesh + fit RMS + valid-depth fraction in the WebSocket JSON ("depth")
```

Why this design (from an architecture review):

- **Scale about the camera origin** leaves every pixel of the 2-D overlay unchanged, so the
  anchor fixes metric depth without fighting the RGB model. The scale is EMA-filtered because a
  body does not change size.
- **Projective ICP** (KinectFusion-style, no KD-tree) uses the depth image for correspondences,
  Huber weights, front-facing visible vertices only, and an optional `cloth_mm` correction for
  gowns/blankets. A hinge-style "one-sided" loss was rejected: it cannot see motion toward the camera.
- **Do not** snap vertices to the cloud (bakes ToF noise into the mesh), stream the cloud to the
  browser, or rely on Azure-Kinect body-tracking joints (redundant, weak supine/under a gown).
- For radiation-oncology style 6DOF, depth-to-reference-surface registration (`roi_icp_cloud`) is
  the measurement to report; SAM 3D vertex jitter is centimetre-level.

Files: `depth/synth.py` (software depth renderer, the test oracle), `depth/anchor.py`,
`depth/icp.py`, `depth/preprocess.py` (flying pixels, erosion, couch-plane removal),
`depth/fusion.py`, `depth/source.py` (`FemtoBoltSource`, `RecordedSource`, `SyntheticSource`).

```bash
python webcam_server.py --depth femto --cloth-mm 0          # live Femto Bolt
python webcam_server.py --depth recorded:session.npz        # npz: depth (T,H,W) uint16 mm, K (3,3), ts_us (T,)
```

Verified (synthetic, `pytest tests/test_depth.py`): scale recovery, 6DOF recovery to a few mm with
noise, flying pixels and a couch plane, gown-offset compensation, end-to-end fusion.

**NOT verified, no hardware was available:**

- `FemtoBoltSource` (pyorbbecsdk names follow the v1.x examples; check your version).
  Use NFOV unbinned mode; the 1 MP wide mode has a lower frame rate. Check the mode/fps table.
- The SAM 3D Body to camera-frame convention in `depth.fusion.mhr_to_camera_mm(flip_yz=True)`
  mirrors the browser's `(t0, -t1, -t2)` placement. Confirm that fused overlay and depth agree.
- Time sync: fit depth frame *k* to RGB frame *k* using device timestamps; never "latest" depth.
- ToF failure modes to expect: flying pixels at silhouettes, multipath under the arms and in the
  couch concavity, invalid pixels on black hair/clothing, warm-up drift of a few mm over tens of minutes.

## Anatomy atlas (TotalSegmentator structures on the MHR mesh)

`atlas/atlas.json` + `atlas/atlas.bin` hold **114 TotalSegmentator v2 structures** (about 140k
triangles, 1.8 MB) already expressed in the MHR rest frame and bound to the MHR skin mesh. Switch it on in the
**Anatomy (atlas)** panel (webcam page) or **Anatomy** section (mobile page).

How it is built (`python -m anatomy.atlas_build <subject_dir> mhr_rest.npz atlas/atlas`):

1. Ground-truth 117-class masks of one subject of the public TotalSegmentator dataset
   (v2.0.1, Zenodo 10047292, subject `s1159`, head to upper thigh). Your own CT segmented with
   TotalSegmentator works the same way.
2. Masks to meshes: marching cubes, Taubin smoothing, quadric decimation (budget by volume).
3. CT skin (threshold + opening, couch removed) registered to the MHR rest mesh: femoral/humeral
   head landmarks (RANSAC sphere fit) similarity, then three rounds of thin-plate-spline warps
   on torso + head skin. The same volumetric warp carries every interior structure.
4. Limb bones are pose-normalised (the CT has the arms down, MHR is a T-pose): humeri and femora are
   swung about their joint centre onto the MHR limb axis.
5. **Containment**: the CT is supine and MHR is standing, so a skin-only warp leaves the spine and
   ribs touching the back (12.7% of vertices outside the skin, median rib/vertebra depth 6.5 mm).
   A smooth volumetric field then pushes every structure to a class-dependent minimum depth below
   the skin (0.0% outside afterwards; ribs >= 10 mm, spine >= 22 mm where possible).
6. The rig (`anatomy/rig.py`) is bound: nine rigid segments, a vertebra chain, rib and girdle
   attachments, and soft-tissue skinning weights.

### Biomechanics (how the anatomy moves)

An earlier version fitted every structure rigidly to its own patch of skin. Neighbouring ribs,
vertebrae and organs then disagreed and the anatomy looked sheared. It is now a kinematic model
(`anatomy/runtime.py` is the reference, `anatomy/anatomy.js` the port; the two agree to float32 rounding):

- **Segments are rigid.** pelvis, lumbar, lower thorax, upper thorax, head, two upper arms, two thighs. Each gets a
  rest-to-live rigid transform (Horn / Kabsch) from a fixed patch of MHR skin vertices. The thorax patches are
  narrow so that armpit skin does not leak arm motion into the trunk.
- **Every bone is rigid.** Hip bones, sacrum, skull and limb bones follow their segment.
- **The spine is a kinematic chain.** Each vertebra's rotation is a quaternion blend of the segments at its height, so
  bending and axial twist are distributed over the vertebrae (swing, twist and bend all fall out of the
  rotation blend). Positions come from forward kinematics, so vertebra spacing is preserved. A smoothed skin-driven
  offset (sigma = 3 vertebrae) keeps the chain placed under the skin.
- **Ribs, sternum, cartilages, clavicles and scapulae ride on the vertebrae they articulate with** (rib k on T-k:
  the costovertebral joint stays exactly rigid).
- **Soft tissue** (organs, vessels, muscles, airway, brain) is skinned between the chain frames bracketing its
  height, so it deforms continuously; gluteal and psoas muscles blend onto the thigh below the hip.

**Validation** (`tests/validate_rig.py`, real MHR articulations from `mhr-body.onnx`, ~2500 sample points; the old
per-structure fit is rebuilt on the same geometry for a fair comparison):

| pose | points pushed outside the skin | change in vertebra/rib spacing (max) | adjacent-vertebra rotation (max) |
|---|---|---|---|
| spine bend / twist / mix | 0.0 - 0.4% (old: 0.0 - 0.1%) | 1.7 - 2.8 mm (old: 5 - 8.5 mm) | 8 - 10 deg (old: 8 - 10 deg) |
| neck and head | 0.0% | 1.6 mm (old: 6.0 mm) | 10.4 deg (old: 22.4 deg) |
| right arm raised | 5.7% (old: 5.2%) | 1.8 mm (old: **36.8 mm**) | 2.8 deg (old: **18.9 deg**) |
| whole-body rotation | 0.0% | 0.0 mm | 0.0 deg |

The old fit stayed inside the skin only because each structure chased its own skin patch; it did not keep a
coherent skeleton (vertebra spacing off by up to 37 mm, neighbouring vertebrae rotating 19 deg apart under a raised arm).
`pytest tests/` checks the rig on controlled segment motions: bones are exactly rigid (< 0.1 mm), a spine twist
distributes monotonically from pelvis to head, vertebra spacing is preserved to 4 mm, ribs keep their vertebra
exactly, and a whole-body rigid motion moves every structure, soft tissue included.

What the model does **not** do:

- No organ physics: no sliding, no breathing, no gravity, no contact. Soft tissue is skinned, not simulated.
- The thorax is rigid per segment. Real ribs flex a little, and the scapula slides over the ribcage.
- The arms: only the humerus (pose-normalised) and the proximal femur are in the atlas, so wrist/elbow/knee motion is not modelled.
- With the arm raised high, MHR's own neck skin collapses inward and cervical vertebrae C4-C6 end up outside the skin (5.7% of
  points). The old fit shows the same artefact: it comes from the MHR surface, not the rig.
- The rig only sees MHR skin motion. A deep motion that does not show on the skin (a rib cage expanding with breath, a
  pelvis tilting under soft tissue) is invisible to it.

### Limits that must stay visible

- **Not patient-specific, not for clinical use.** Internal positions are an atlas guess scaled to
  the skin. Expected localisation error is roughly 1-2 cm for bones near the skin and several cm
  for deep soft tissue (`UNCERTAINTY_MM` in `anatomy/classes.py` is a prior, not a measurement).
- The source CT stops at the upper thigh and has no arms: **no distal limbs**, and 3 of the 117
  classes are empty in this subject (`kidney_cyst_*`, `prostate`). Distal anatomy would need another
  source (for example BodyParts3D / Z-Anatomy, both CC BY-SA share-alike).
- One atlas subject, one body habitus. The build should be repeated for several subjects, and
  validated leave-one-out (warp subject A's atlas to subject B's skin; score Dice/ASSD per class).
- Arms: the CT arms are not in the MHR T-pose, so the registration is driven by torso and head only.
- TotalSegmentator **dataset** is CC BY 4.0: cite Wasserthal et al., *Radiology: Artificial
  Intelligence* 2023. The TotalSegmentator code is Apache-2.0, but some model subtasks are
  restricted for commercial use. MHR / SAM 3D Body / InstantHMR files are under the SAM License:
  read it before redistributing the atlas or the MHR rest mesh.

### Tests

```bash
pip install numpy scipy pytest scikit-image nibabel trimesh fast-simplification rtree
pytest tests/                                  # 18 tests: depth, rig invariants on the shipped atlas, JS-vs-numpy parity (needs node)
PYTHONPATH=. python tests/validate_rig.py atlas/atlas - mhr-body.onnx mhr_rest.npz   # real-MHR-pose comparison (needs the onnx file)
```

## Troubleshooting

### First inference times out

A new `torch.compile` graph may take substantially longer on first use. Increase the WebSocket inference timeout and retry after compilation completes.

### `Unable to load EGL library`

Reinstall the generic EGL loader:

```bash
sudo apt-get install --reinstall libegl1
```

### Triton reports an unsupported CUDA version

Force Triton to use the Conda CUDA 12.4 assembler:

```bash
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
```

### `ModuleNotFoundError: flask_sock`

Install flask-sock with the active Conda Python:

```bash
python -m pip install flask-sock
```

### High or variable preprocessing latency

Confirm the thread limits are active:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

The uncapped environment used 128 intra-op and 128 inter-op threads and produced substantially higher latency.

## Privacy

Webcam frames are transmitted to the configured GPU server for inference.

Before using identifiable, clinical, research, or otherwise sensitive video, confirm that the server, browser connection, storage behavior, and network path comply with applicable institutional privacy and data-handling requirements.
