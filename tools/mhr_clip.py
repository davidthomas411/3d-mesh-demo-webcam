"""Motion-capture clips for the rig: load per-frame MHR model parameters (e.g. SAM 3D Body output) and prepare them for the atlas.

Clip file (.npz): `mhr_model_params` (T, 204) (or (T, 1, 204)): exactly the `mhr_model_params` field SAM 3D Body returns per person
(layout: [0:3] global translation*10, [3:6] global rotation (euler xyz), [6:136] body pose, [136:204] scale params);
optional `fps`, `source`. Also accepts a .json list of per-frame 204-vectors.

prepare(): unwrap + temporal smoothing, then neutralise what the atlas cannot use: the camera-frame global translation/rotation
(the patient lies in the CT frame; only articulation matters) and the per-person scale parameters (the atlas is the mean-shape MHR).
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

N_PARAMS = 204


def load_clip(path):
    if path.endswith(".json"):
        arr = np.asarray(json.load(open(path)), np.float32); fps = 30.0; source = path
    else:
        z = np.load(path, allow_pickle=True)
        arr = np.asarray(z["mhr_model_params"], np.float32)
        fps = float(z["fps"]) if "fps" in z.files else 30.0
        source = str(z["source"]) if "source" in z.files else path
    arr = arr.reshape(len(arr), -1)
    if arr.shape[1] != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} MHR model parameters per frame, got {arr.shape[1]}")
    return arr, fps, source


def prepare(params, smooth_frames=1.5, neutral_root=True, neutral_scale=True):
    p = np.unwrap(params.astype(np.float64), axis=0)                # joint angles are radians: remove 2*pi jumps before filtering
    if smooth_frames > 0:
        p = gaussian_filter1d(p, smooth_frames, axis=0, mode="nearest")
    if neutral_root:
        p[:, 0:6] = 0.0
    if neutral_scale:
        p[:, 136:] = 0.0
    return p.astype(np.float32)
