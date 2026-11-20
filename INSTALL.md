# Installing the local demo on another PC

This is the set-up for the real-time webcam demo on the `local-anatomy` branch: webcam -> Fast SAM 3D Body -> body mesh with a
skeleton inside it, a DRR and a CT view, and reference capture with shifts. It was worked out on Windows 11 with one NVIDIA GPU
(RTX 4060, 8 GB). `requirements-local.txt` lists the exact package versions of that working install, as a reference.

What you get depends on which data files are present. None of them are in git.

| On the PC | What the page shows |
|---|---|
| Code only (this repo + Fast SAM 3D Body) | Body mesh, CT-atlas bones, surface DRR, label slice, reference capture and shifts |
| + SKEL atlas (`skel.*`, 3 MB) | The SKEL skeleton (whole body, hands, feet, jaw). This is the default skeleton |
| + CT volume (`ct_volume.*`, 60-70 MB per subject) | The real CT slice and the CT-voxel DRR |
| + PET volume (`ct_pet.*`, 4 MB per subject) | The PET checkbox |

So for SKEL without CT or PET, do sections 1 to 5 and stop.

## 0. The branch

The repository's default branch is `main`, which does not have this work. After cloning:

```bash
git checkout local-anatomy
```

## 1. Folder layout

Everything sits side by side in one folder. The scripts find each other by these relative paths.

```text
<work>/
  3d-mesh-demo-webcam/     this repository, branch local-anatomy
  Fast-SAM-3D-Body/        upstream model code, its .venv and checkpoints
  body_models/             SKEL model files and the fit (only needed to rebuild the SKEL atlas)
  ct_data/                 CT / PET subjects (only needed to rebuild CT volumes)
```

```bash
git clone https://github.com/davidthomas411/3d-mesh-demo-webcam.git
```

```bash
git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git
```

The working install is at Fast-SAM-3D-Body commit `d72aa36`.

## 2. Python environment (inside Fast-SAM-3D-Body)

Python 3.11, an NVIDIA driver with CUDA 12.x support. Create the environment in `Fast-SAM-3D-Body/.venv` (that is where
`run_local.cmd` looks for it):

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

Then the upstream dependencies. `Fast-SAM-3D-Body/setup_env.sh` is the authoritative list (it is written for conda on Linux; on
Windows run its `pip install` lines with `.venv\Scripts\python -m pip`). Detectron2 is not needed: the demo uses the YOLO
detector. The parts this demo relies on:

```bash
.venv\Scripts\python -m pip install ultralytics tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnx onnxruntime-gpu git+https://github.com/microsoft/MoGe.git
```

And what this repository adds:

```bash
.venv\Scripts\python -m pip install Flask flask-sock simple-websocket numpy scipy pytest scikit-image nibabel trimesh fast-simplification rtree
```

If an import is still missing when the server starts, install the version listed in `requirements-local.txt`.

## 3. Model files and the TensorRT engine

The SAM 3D Body checkpoint (`checkpoints/sam-3d-body-dinov3/`, 2 GB) is downloaded from Hugging Face on the first run; the model
page may ask you to log in and accept its terms first. The YOLO pose weights go in `checkpoints/yolo/yolo11m-pose.pt`
(ultralytics downloads them by name).

Two local changes to Fast-SAM-3D-Body are kept in `local_setup/` here, because the upstream scripts are fixed at a 512 input and
the demo runs at 384:

```bash
copy ..\3d-mesh-demo-webcam\local_setup\build_backbone_trt_local.py .
```

```bash
git apply ..\3d-mesh-demo-webcam\local_setup\dinov3_tensorrt_dynamic_size.patch
```

Build the backbone engine. An engine is specific to the GPU and TensorRT version, so build it on each PC rather than copying it
(several minutes):

```bash
.venv\Scripts\python build_backbone_trt_local.py 384
```

It should end with `ENGINE_OK` and leave `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16_384.engine`.

## 4. The rest mesh

Several tools need the neutral body (`mhr_rest.npz`: vertices, faces, rest keypoints). It is made from the checkpoint, once:

```bash
.venv\Scripts\python ..\3d-mesh-demo-webcam\tools\make_mhr_rest.py checkpoints\sam-3d-body-dinov3\assets\mhr_model.pt checkpoints\sam-3d-body-dinov3\model.ckpt mhr_rest.npz
```

## 5. SKEL

SKEL (Keller et al. 2023) is under a non-commercial research licence, so neither its model files nor the atlas built from them
are in git. There are two ways to get it onto a PC.

### 5a. Copy the built atlas (quickest)

The page only needs the built atlas, 3 MB in all. Copy these files from a PC that has them, keeping the paths:

```text
3d-mesh-demo-webcam/atlas/skel.json
3d-mesh-demo-webcam/atlas/skel.bin
3d-mesh-demo-webcam/atlas/subjects/e0279/skel.json
3d-mesh-demo-webcam/atlas/subjects/e0279/skel.bin
3d-mesh-demo-webcam/atlas/subjects/s1159/skel.json
3d-mesh-demo-webcam/atlas/subjects/s1159/skel.bin
```

Nothing else is needed: no SKEL model files, no extra Python package.

### 5b. Build it from the SKEL model

1. Download `skel_models_v1.1.zip` from https://skel.is.tue.mpg.de (free account, licence to accept) and unzip it to
   `body_models/skel_models_v1.1` (it must contain `skel_male.pkl`).
2. Download `mhr2smpl_mapping.npz` from https://github.com/facebookresearch/MHR (`tools/mhr_smpl_conversion/assets`) into
   `body_models/`.
3. Install the SKEL package, without its dependencies (they pin old versions):

```bash
.venv\Scripts\python -m pip install git+https://github.com/MarilynKeller/SKEL --no-deps
```

4. Run the build from the `3d-mesh-demo-webcam` folder:

```bash
local_setup\build_skel.cmd
```

That fits SKEL to the neutral body (a few minutes on CPU) and writes `skel.*` next to every `atlas.*` (the default one and each
reference subject).

## 6. Run

From the `3d-mesh-demo-webcam` folder, in a terminal of your own (it stays up until Ctrl-C):

```bash
run_local.cmd
```

Wait for `ESTIMATOR_READY`, then open http://127.0.0.1:8097 in Chrome and press "Start webcam and inference". The camera only
works on `localhost` / `127.0.0.1` (or https), not on the PC's network address.

Check: the status line should show inference around 110-150 ms on an RTX 4060; the 3-D view's gear menu should offer
"SKEL (learned, whole skeleton)" under Skeleton. If it only offers the CT atlas bones, the `skel.*` files are not where the page
looks for them (section 5a).

The tests do not need a GPU:

```bash
..\Fast-SAM-3D-Body\.venv\Scripts\python -m pytest tests -q
```

## 7. Optional: the CT and PET volumes

Without them the CT panel shows the flat label slice and the DRR is made from bone surfaces only; everything else works.

Quickest is again to copy the built files from a PC that has them (about 140 MB for both subjects):

```text
3d-mesh-demo-webcam/atlas/subjects/<id>/ct_volume.json  ct_volume.bin
3d-mesh-demo-webcam/atlas/subjects/<id>/ct_pet.json     ct_pet.bin        (only for the PET checkbox)
```

To build them from the scans instead, see "Real CT, deformed live" in `README.md`. The PET-CT subject (ENHANCE.PET 0279) is
CC BY-NC 4.0.

## Troubleshooting

- **Old page, no gear menus, no chooser.** The checkout is on `main`. `git checkout local-anatomy`, `git pull`.
- **"Site can't be reached" / "WebSocket connection failed".** The server is not running. Start `run_local.cmd` and leave that
  terminal open.
- **`run_local.cmd` cannot find Python or the engine.** It expects `..\Fast-SAM-3D-Body\.venv` and the 384 engine of section 3.
- **Engine fails to load.** It was copied from another PC or built with another TensorRT version. Rebuild it (section 3).
- **Latency of a second or more.** Something else is using the GPU, often a second browser tab with the page open.
- **Several people in view.** The server follows the person it was already on, else the largest one near the centre.
