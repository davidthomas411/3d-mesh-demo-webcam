import argparse
import contextlib
import json
import os
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_sock import Sock

from mocap.core.setup_estimator import build_default_estimator

# ---- local real-time settings (all optional, set through the environment) ----
# IMG_SIZE        model input size (512 upstream default; 384 with a TensorRT backbone is the fast setting)
# YOLO_PATH       person/pose detector weights
# TRACK_BOX=1     take the person box from the previous mesh instead of running the detector on every frame
# DETECT_EVERY    with TRACK_BOX, run the detector every N frames only to confirm the person is still there
# QUIET=1         drop the estimator's per-frame debug prints (slow on a Windows console)
MODEL_IMG_SIZE = int(os.environ.get("IMG_SIZE", "512"))
YOLO_PATH = os.environ.get("YOLO_PATH", "checkpoints/yolo/yolo11m-pose.pt")
TRACK_BOX = os.environ.get("TRACK_BOX", "0") == "1"
DETECT_EVERY = int(os.environ.get("DETECT_EVERY", "30"))
QUIET = os.environ.get("QUIET", "0") == "1"

if QUIET:
    # upstream empties the CUDA cache on every frame; with one model and fixed shapes that only costs time
    torch.cuda.empty_cache = lambda: None


class _NullWriter:
    def write(self, _text):
        return 0

    def flush(self):
        pass


def box_from_mesh(vertices, cam_t, focal, width, height):
    """Image-space bounding box of the predicted mesh (pinhole, principal point at the image centre)."""
    X = vertices + np.asarray(cam_t, dtype=np.float32).reshape(1, 3)
    z = np.maximum(X[:, 2], 1e-3)
    u = focal * X[:, 0] / z + width / 2.0
    v = focal * X[:, 1] / z + height / 2.0
    box = np.array([max(0.0, u.min()), max(0.0, v.min()), min(width - 1.0, u.max()), min(height - 1.0, v.max())], dtype=np.float32)
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return box.reshape(1, 4)


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
    b = np.asarray(b, dtype=np.float32).reshape(-1)[:4]
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return float(iw * ih / union) if union > 0 else 0.0

# One person only. The page shows one body, so when the detector finds several the model must run on one of them (it used to run
# on all of them and the frame was then thrown away). The person followed on the previous frame is kept (largest overlap with
# their box); with nobody followed yet, the largest box near the middle of the image wins. The detector wrapper does not return
# confidences, so size and position stand in for "most confident".
person_pick = {"last": None, "seen": 0}


def pick_person(boxes, image_shape, last=None):
    """Index of the box to keep. boxes: (n, 4) x0, y0, x1, y1."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) <= 1:
        return 0
    if last is not None:
        overlap = [box_iou(b, last) for b in boxes]
        if max(overlap) > 0.2:
            return int(np.argmax(overlap))
    h, w = image_shape[:2]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
    off = np.hypot(((boxes[:, 0] + boxes[:, 2]) / 2 - w / 2) / (w / 2), ((boxes[:, 1] + boxes[:, 3]) / 2 - h / 2) / (h / 2))
    return int(np.argmax(area * (1.0 - 0.5 * np.clip(off, 0.0, 1.0))))


def single_person_detector(detector):
    """Make detector.run_human_detection return at most one person (see pick_person)."""
    run = detector.run_human_detection

    def run_one(img, **kwargs):
        result = run(img, **kwargs)
        boxes = result["boxes"] if isinstance(result, dict) else result
        n = 0 if boxes is None else len(boxes)
        person_pick["seen"] = n
        if n <= 1:
            return result
        i = pick_person(np.asarray(boxes)[:, :4], img.shape, person_pick["last"])
        if isinstance(result, dict):
            return {k: (v[i:i + 1] if v is not None and hasattr(v, "__len__") and len(v) == n else v) for k, v in result.items()}
        return boxes[i:i + 1]

    detector.run_human_detection = run_one


# Optional Femto Bolt / depth fusion (see depth/). Off unless --depth is given.
depth_source = None
depth_fusion = None

app = Flask(__name__)
sock = Sock(app)

state_lock = threading.Lock()
frame_event = threading.Event()
mesh_condition = threading.Condition(state_lock)

latest_frame = None
latest_frame_owner = None   # which connection sent latest_frame (box tracking is per camera stream)
latest_mesh = None
frame_id = 0
mesh_id = 0
dropped_frames = 0
last_inference_seconds = None
last_error = None
estimator_ready = False
faces = None
FOCAL_CALIBRATION_FRAMES = 30

# Focal calibration belongs to a camera, so it is kept per stream (per WebSocket connection, or "http" for the
# POST /frame route): two browsers, or a webcam and a video file, no longer share one median.
calibrations = {}
last_calibration_owner = None


def calibration_for(owner):
    """Calibration state of one stream. Call with state_lock held."""
    state = calibrations.get(owner)
    if state is None:
        state = calibrations[owner] = {
            "history": [],             # focal lengths estimated so far
            "cam": None,               # fixed intrinsics once calibrated (torch tensor)
            "mode": "fixed",           # "fixed": use cam when available; "dynamic": estimate every frame
            "frames": FOCAL_CALIBRATION_FRAMES,
            "fixed_when_ready": False,  # set by "recalibrate": go back to fixed as soon as the median is in
        }
    return state

runtime_config = {
    "inference_mode": "body",
    "intrinsics_mode": "fixed",
    "calibration_frames": 30,
}


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def inference_worker():
    global latest_frame
    global latest_mesh
    global mesh_id
    global dropped_frames
    global last_inference_seconds
    global last_error
    global estimator_ready
    global faces
    global runtime_config
    global depth_fusion
    global last_calibration_owner

    print("Loading persistent Fast SAM 3D Body estimator...", flush=True)

    estimator = build_default_estimator(
        image_size=MODEL_IMG_SIZE,
        yolo_model_path=YOLO_PATH,
        fov_model_size="s",
        fov_resolution_level=0,
        fov_fixed_size=512,
        fov_fast_mode=True,
    )

    if getattr(estimator, "detector", None) is not None:
        single_person_detector(estimator.detector)
    faces = np.asarray(estimator.faces, dtype=np.int32)
    if depth_source is not None:
        from depth.fusion import DepthFusion
        depth_fusion = DepthFusion(faces, **depth_fusion_args)
    estimator_ready = True

    print(
        f"ESTIMATOR_READY vertices=18439 faces={len(faces)} "
        f"img_size={MODEL_IMG_SIZE} track_box={TRACK_BOX} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    quiet_out = _NullWriter()
    tracked_box = None          # person box derived from the previous mesh
    tracked_owner = None
    frames_since_detect = 0
    inference_window = []

    while True:
        frame_event.wait()
        frame_event.clear()

        with state_lock:
            item = latest_frame
            item_owner = latest_frame_owner
            latest_frame = None

        if item_owner != tracked_owner:
            # a different camera stream: its person box has nothing to do with the last one
            tracked_owner = item_owner
            tracked_box = None
            person_pick["last"] = None

        if item is None:
            continue

        current_frame_id, client_timestamp_ms, encode_ms, server_receive_ns, server_receive_wall_ms, jpeg_bytes = item
        worker_take_ns = time.perf_counter_ns()
        queue_wait_ms = (worker_take_ns - server_receive_ns) / 1e6

        decode_start_ns = time.perf_counter_ns()
        image = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            with state_lock:
                last_error = "JPEG decode failed"
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1e6

        start = time.perf_counter()

        try:
            with state_lock:
                inference_mode = runtime_config["inference_mode"]
                calibration = calibration_for(item_owner)
                last_calibration_owner = item_owner
                if calibration.get("bolt") and not calibration.get("factory"):
                    # the webcam is the Femto Bolt's colour camera: its factory intrinsics, not an estimate from the picture
                    k = cloud_source.info.get("colour_k") if cloud_source is not None else None
                    if k:
                        height, width = image_rgb.shape[:2]
                        calibration["cam"] = torch.tensor([[[k[0] * width, 0.0, k[2] * width], [0.0, k[1] * height, k[3] * height],
                                                           [0.0, 0.0, 1.0]]], dtype=torch.float32)
                        calibration["mode"] = "fixed"; calibration["factory"] = True
                        print(f"FACTORY_INTRINSICS focal={k[0] * width:.1f} size={width}x{height}", flush=True)
                intrinsics_mode = calibration["mode"]
                active_cam_intrinsics = (
                    calibration["cam"]
                    if intrinsics_mode == "fixed"
                    else None
                )

            # Box tracking: the detector costs about 18 ms a frame. Once a person is found, the next
            # frame's box comes from the previous mesh. The detector still runs every DETECT_EVERY
            # frames, only to confirm that somebody is there and that the tracked box is on them.
            use_tracked = (
                TRACK_BOX
                and inference_mode == "body"
                and tracked_box is not None
                and frames_since_detect < DETECT_EVERY
            )

            with contextlib.redirect_stdout(quiet_out) if QUIET else contextlib.nullcontext():
                outputs = estimator.process_one_image(
                    image_rgb,
                    bboxes=tracked_box if use_tracked else None,
                    cam_int=active_cam_intrinsics,
                    inference_type=inference_mode,
                    hand_box_source="yolo_pose",
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if use_tracked:
                frames_since_detect += 1
            else:
                frames_since_detect = 0

            if len(outputs) != 1:
                tracked_box = None
                person_pick["last"] = None
                with mesh_condition:
                    mesh_id += 1

                    latest_mesh = {
                        "mesh_id": mesh_id,
                        "source_frame_id": current_frame_id,
                        "no_person": True,
                        "client_timestamp_ms": client_timestamp_ms,
                        "encode_ms": float(encode_ms),
                        "queue_wait_ms": queue_wait_ms,
                        "decode_ms": decode_ms,
                    }

                    last_inference_seconds = elapsed
                    last_error = None
                    mesh_condition.notify_all()

                print(
                    f"NO_PERSON frame={current_frame_id} "
                    f"detected={len(outputs)} "
                    f"inference={elapsed:.3f}s",
                    flush=True,
                )
                continue

            focal_length = float(
                np.asarray(outputs[0]["focal_length"]).reshape(-1)[0]
            )
            focal_length_history = calibration["history"]
            focal_length_history.append(focal_length)
            if len(focal_length_history) > 300:
                del focal_length_history[:-300]

            if (
                calibration["cam"] is None
                and len(focal_length_history) >= calibration["frames"]
            ):
                calibrated_focal = float(
                    np.median(
                        focal_length_history[
                            :calibration["frames"]
                        ]
                    )
                )
                height, width = image_rgb.shape[:2]
                calibration["cam"] = torch.tensor(
                    [[
                        [calibrated_focal, 0.0, width / 2.0],
                        [0.0, calibrated_focal, height / 2.0],
                        [0.0, 0.0, 1.0],
                    ]],
                    dtype=torch.float32,
                )
                if calibration["fixed_when_ready"]:
                    # a recalibration is finished: stop estimating the focal length on every frame
                    calibration["mode"] = "fixed"
                    calibration["fixed_when_ready"] = False
                print(
                    f"FOCAL_CALIBRATED frames={calibration['frames']} "
                    f"focal={calibrated_focal:.3f} "
                    f"size={width}x{height}",
                    flush=True,
                )

            vertices = to_numpy(outputs[0]["pred_vertices"]).astype(
                np.float32,
                copy=False,
            )

            if vertices.shape != (18439, 3):
                raise RuntimeError(
                    f"Unexpected vertex shape: {vertices.shape}"
                )

            # MHR skeleton for the rig: 127 joint positions (same frame as the vertices) and global rotations.
            # The rotations come in the MHR model frame; the vertices are flipped (x, -y, -z), so R -> F R F.
            joints = None
            try:
                joint_pos = to_numpy(outputs[0]["pred_joint_coords"]).reshape(-1, 3)
                joint_rot = to_numpy(outputs[0]["pred_global_rots"]).reshape(-1, 3, 3).copy()
                joint_rot[:, 0, 1:] *= -1.0
                joint_rot[:, 1:, 0] *= -1.0
                if len(joint_pos) == len(joint_rot):
                    joints = np.concatenate([joint_pos.reshape(-1), joint_rot.reshape(-1)]).astype("<f4")
            except (KeyError, TypeError, ValueError):
                joints = None

            if "bbox" in outputs[0]:
                person_pick["last"] = to_numpy(outputs[0]["bbox"]).reshape(-1)[:4].copy()      # the person to stay with
            if TRACK_BOX:
                image_height, image_width = image_rgb.shape[:2]
                tracked_box = box_from_mesh(
                    vertices,
                    to_numpy(outputs[0]["pred_cam_t"]).reshape(-1),
                    focal_length,
                    image_width,
                    image_height,
                )
                if not use_tracked and tracked_box is not None and "bbox" in outputs[0]:
                    # detector frame: if the mesh box is not on the detected person, start again
                    if box_iou(tracked_box, to_numpy(outputs[0]["bbox"])) < 0.3:
                        tracked_box = None

            depth_info = None
            if depth_source is not None and depth_fusion is not None:
                try:
                    from depth.fusion import mhr_to_camera_mm
                    depth_frame, depth_ts_us = depth_source.read()
                    if depth_frame is not None:
                        cam_t_np = to_numpy(outputs[0]["pred_cam_t"]).reshape(-1)
                        v_mm = mhr_to_camera_mm(vertices, cam_t_np)
                        fused_mm, depth_info = depth_fusion.process(v_mm, depth_frame)
                        depth_info["device_ts_us"] = depth_ts_us
                        if depth_info.get("ok"):
                            # report the metric camera translation: centroid shift in metres
                            shift = (fused_mm.mean(0) - v_mm.mean(0)) / 1000.0
                            depth_info["cam_t_shift_m"] = (shift * np.array([1.0, -1.0, -1.0])).tolist()
                except Exception as exc:  # depth must never take the RGB pipeline down
                    depth_info = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            if depth_info is None and calibration.get("depth_fusion"):
                # Femto Bolt: this mesh becomes the shape and pose prior of the low-latency path, which fits it to every depth
                # frame and sends the fitted mesh itself (depth/tracker.py). The page does not show this one while that runs.
                try:
                    if not calibration.get("bolt"):
                        depth_info = {"ok": False, "reason": "the webcam is not the Femto Bolt's colour camera"}
                    else:
                        lf = get_live_fit()
                        cam_t_mm = to_numpy(outputs[0]["pred_cam_t"]).reshape(-1).astype(np.float64) * 1000.0
                        jp = jr = None
                        if joints is not None:
                            nj = len(joints) // 12
                            jp = joints[:nj * 3].reshape(nj, 3).astype(np.float64) * 1000.0 + cam_t_mm
                            jr = joints[nj * 3:].reshape(nj, 3, 3).astype(np.float64)
                        lf.set_model(item_owner, vertices.astype(np.float64) * 1000.0 + cam_t_mm, jp, jr, mesh_id + 1,
                                     {"cam_t_mm": cam_t_mm, "focal_length": focal_length, "client_timestamp_ms": client_timestamp_ms})
                        last = lf.result
                        depth_info = {"ok": lf.active(item_owner), "fit": True, "factory_intrinsics": bool(calibration.get("factory")),
                                      "reason": None if lf.active(item_owner) else (cloud_source.error or "no fit yet: is the body in the depth image?")}
                        if last is not None:
                            depth_info.update(scale=last["info"].get("scale"), median_mm=last["info"].get("median_mm"))
                except Exception as exc:  # depth must never take the RGB pipeline down
                    depth_info = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

            if recording["left"] > 0:
                # Recorder (GET /record): the picture, what the model made of it, and the Femto Bolt depth frame of the moment,
                # for working on the mesh's placement offline (tools/analyse_depth_recording.py).
                try:
                    cam = active_cam_intrinsics
                    dframe = get_cloud().get()
                    recording["frames"].append({
                        "jpeg": bytes(jpeg_bytes), "verts": vertices.astype(np.float32),
                        "cam_t": to_numpy(outputs[0]["pred_cam_t"]).reshape(-1).astype(np.float32),
                        "focal": focal_length, "K": None if cam is None else cam.reshape(3, 3).numpy().astype(np.float32),
                        "kp2d": to_numpy(outputs[0]["pred_keypoints_2d"]).astype(np.float32),
                        "depth": None if dframe is None else dframe[1], "depth_seq": 0 if dframe is None else dframe[0],
                        "joints": joints,
                        "t": time.time(), "factory": bool(calibration.get("factory")), "anchor": depth_info,
                    })
                    recording["left"] -= 1
                    if recording["left"] == 0:
                        _save_recording()
                except Exception as exc:
                    recording.update(left=0, error=f"{type(exc).__name__}: {exc}")

            with state_lock:
                mesh_id += 1
                latest_mesh = {
                    "depth": depth_info,
                    "mesh_id": mesh_id,
                    "source_frame_id": current_frame_id,
                    "no_person": False,
                    "vertices": vertices,
                    "joints": joints,
                    "server_receive_wall_ms": server_receive_wall_ms,
                    "encode_ms": float(encode_ms),
                    "queue_wait_ms": queue_wait_ms,
                    "decode_ms": decode_ms,
                    "pred_keypoints_2d": to_numpy(outputs[0]["pred_keypoints_2d"]),
                    "pred_cam_t": to_numpy(outputs[0]["pred_cam_t"]).reshape(-1),
                    "focal_length": focal_length,
                    "client_timestamp_ms": client_timestamp_ms,
                    "intrinsics_mode": calibration["mode"],
                    "calibrated": calibration["cam"] is not None,
                    "calibration_samples": len(focal_length_history),
                }
                last_inference_seconds = elapsed
                last_error = None
                mesh_condition.notify_all()

            inference_window.append(elapsed * 1000.0)
            if len(inference_window) >= 60:
                window = np.asarray(inference_window)
                print(
                    f"MESH {mesh_id} inference median={np.median(window):.1f} ms "
                    f"p90={np.percentile(window, 90):.1f} ms "
                    f"tracked_box={use_tracked}",
                    flush=True,
                )
                inference_window.clear()

        except Exception as exc:
            with state_lock:
                last_error = f"{type(exc).__name__}: {exc}"
            print(f"INFERENCE_ERROR: {last_error}", flush=True)


@app.get("/")
def index():
    return send_from_directory(".", "webcam.html")


# A recorded gantry sweep with its tracking (tools/gantry_export.py). The export holds pictures of a person, so it lives
# outside the repository: GANTRY_WEB, by default ../recordings/gantry/web.
GANTRY_WEB = os.path.abspath(os.environ.get("GANTRY_WEB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recordings", "gantry", "web"))


@app.get("/gantry")
def gantry_page():
    return send_from_directory(".", "gantry.html")


@app.get("/gantry-data/<path:name>")
def gantry_data(name):
    if not (name in ("index.json", "replay.mp4", "replay.webm", "replay_pose.json") or (name.startswith("f/") and name.endswith(".jpg"))):
        return Response("Not found", status=404)
    return send_from_directory(GANTRY_WEB, name)


@app.get("/anatomy/<name>")
def anatomy_js(name):
    if name not in ("anatomy.js", "stream.js", "xray.js", "ct.js", "ctvol.js"):
        return Response("Not found", status=404)
    return send_from_directory("anatomy", name, mimetype="text/javascript")


@app.get("/atlas/<name>")
def atlas_file(name):
    if name not in ("atlas.json", "atlas.bin", "skel.json", "skel.bin", "ct_volume.json", "ct_volume.bin", "ct_pet.json", "ct_pet.bin"):
        return Response("Not found", status=404)
    return send_from_directory("atlas", name)


ATLAS_FILES = ("atlas.json", "atlas.bin", "skel.json", "skel.bin", "ct_volume.json", "ct_volume.bin", "ct_pet.json", "ct_pet.bin")
SUBJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", "subjects")


@app.get("/subjects")
def subjects():
    """Reference CTs that have a rig built: one folder each under atlas/subjects (tools/build_ct_volume.py)."""
    out = []
    if os.path.isdir(SUBJECTS_DIR):
        for sid in sorted(os.listdir(SUBJECTS_DIR)):
            d = os.path.join(SUBJECTS_DIR, sid)
            if not sid.replace("_", "").replace("-", "").isalnum() or not os.path.exists(os.path.join(d, "atlas.json")):
                continue
            try:
                with open(os.path.join(d, "atlas.json"), encoding="utf-8") as f:
                    src = json.load(f).get("source", {})
                pet = None; rigid = None
                if os.path.exists(os.path.join(d, "ct_volume.json")):
                    with open(os.path.join(d, "ct_volume.json"), encoding="utf-8") as f:
                        rigid = bool(json.load(f).get("rigid"))
                if os.path.exists(os.path.join(d, "ct_pet.json")):
                    with open(os.path.join(d, "ct_pet.json"), encoding="utf-8") as f:
                        pet = "simulated" if json.load(f).get("simulated") else "measured"
                out.append({"id": sid, "dataset": str(src.get("dataset", ""))[:80], "subject": str(src.get("subject", sid)),
                            "ct": os.path.exists(os.path.join(d, "ct_volume.bin")), "pet": pet, "rigid": rigid})
            except (OSError, ValueError):
                continue
    return jsonify(out)


@app.get("/atlas/subjects/<sid>/<name>")
def subject_file(sid, name):
    if name not in ATLAS_FILES or not sid.replace("_", "").replace("-", "").isalnum():
        return Response("Not found", status=404)
    return send_from_directory(os.path.join(SUBJECTS_DIR, sid), name)


@app.post("/frame")
def receive_frame():
    global latest_frame
    global latest_frame_owner
    global frame_id
    global dropped_frames

    server_receive_ns = time.perf_counter_ns()
    server_receive_wall_ms = time.time() * 1000
    server_receive_wall_ms = time.time() * 1000
    jpeg_bytes = request.get_data(cache=False)
    client_timestamp_ms = request.headers.get("X-Capture-Timestamp-Ms", "")
    encode_ms = request.headers.get("X-Encode-Ms", "0")

    if len(jpeg_bytes) < 100:
        return Response("Invalid frame", status=400)

    with state_lock:
        frame_id += 1
        if latest_frame is not None:
            dropped_frames += 1
        latest_frame = (frame_id, client_timestamp_ms, encode_ms, server_receive_ns, server_receive_wall_ms, jpeg_bytes)
        latest_frame_owner = "http"
        accepted_frame_id = frame_id

    frame_event.set()

    with mesh_condition:
        completed = mesh_condition.wait_for(
            lambda: (
                latest_mesh is not None
                and latest_mesh["source_frame_id"] == accepted_frame_id
            ),
            timeout=30.0,
        )

        if not completed:
            return Response("Inference timeout", status=504)

        current = latest_mesh

    response = Response(
        current["vertices"].astype("<f4", copy=False).tobytes(),
        content_type="application/octet-stream",
    )
    response.headers["X-Frame-Id"] = str(accepted_frame_id)
    response.headers["X-Mesh-Id"] = str(current["mesh_id"])
    response.headers["X-Source-Frame-Id"] = str(current["source_frame_id"])
    response.headers["X-Inference-Seconds"] = str(last_inference_seconds)
    response.headers["X-Encode-Ms"] = str(current["encode_ms"])
    response.headers["X-Queue-Wait-Ms"] = str(current["queue_wait_ms"])
    response.headers["X-Decode-Ms"] = str(current["decode_ms"])
    response.headers["X-Capture-Timestamp-Ms"] = current["client_timestamp_ms"]
    response.headers["X-Pred-Cam-T"] = ",".join(
        map(str, current["pred_cam_t"])
    )
    response.headers["X-Focal-Length"] = str(current["focal_length"])
    return response


@app.get("/topology")
def topology():
    if not estimator_ready or faces is None:
        return jsonify({"ready": False}), 503

    return jsonify({
        "ready": True,
        "vertex_count": 18439,
        "faces": faces.reshape(-1).tolist(),
    })


@app.get("/mesh")
def mesh():
    requested_id = request.args.get("after", default=0, type=int)

    with state_lock:
        current = latest_mesh
        error = last_error
        inference_seconds = last_inference_seconds

    if current is None or current["mesh_id"] <= requested_id:
        return Response(status=204)

    payload = current["vertices"].astype("<f4", copy=False).tobytes()
    joint_count = 0
    if request.args.get("joints") and current.get("joints") is not None:
        # same layout as the WebSocket message: vertices, then n*3 joint positions, then n*9 rotation entries
        payload += current["joints"].tobytes()
        joint_count = len(current["joints"]) // 12

    response = Response(
        payload,
        content_type="application/octet-stream",
    )
    response.headers["X-Joints"] = str(joint_count)
    response.headers["X-Mesh-Id"] = str(current["mesh_id"])
    response.headers["X-Source-Frame-Id"] = str(current["source_frame_id"])
    response.headers["X-Inference-Seconds"] = str(inference_seconds)
    response.headers["X-Server-Receive-Wall-Ms"] = str(current["server_receive_wall_ms"])
    response.headers["X-Encode-Ms"] = str(current["encode_ms"])
    response.headers["X-Queue-Wait-Ms"] = str(current["queue_wait_ms"])
    response.headers["X-Decode-Ms"] = str(current["decode_ms"])
    response.headers["X-Capture-Timestamp-Ms"] = current["client_timestamp_ms"]
    response.headers["X-Pred-Cam-T"] = ",".join(map(str, current["pred_cam_t"]))
    response.headers["X-Focal-Length"] = str(current["focal_length"])
    return response


@app.get("/focal-stats")
def focal_stats():
    with state_lock:
        state = calibrations.get(last_calibration_owner)
        history = list(state["history"]) if state else []
        is_calibrated = bool(state and state["cam"] is not None)
        stream_count = len(calibrations)
    values = np.asarray(history, dtype=np.float64)

    if values.size == 0:
        return jsonify({"count": 0, "streams": stream_count})

    median = float(np.median(values))

    return jsonify({
        "count": int(values.size),
        "calibrated": is_calibrated,
        "streams": stream_count,
        "median": median,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.ptp(values)),
        "range_percent_of_median": (
            float(np.ptp(values) / median * 100.0)
            if median != 0 else None
        ),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    })


@app.get("/status")
def status():
    with state_lock:
        return jsonify({
            "estimator_ready": estimator_ready,
            "received_frames": frame_id,
            "dropped_frames": dropped_frames,
            "mesh_id": mesh_id,
            "inference_seconds": last_inference_seconds,
            "error": last_error,
            "img_size": MODEL_IMG_SIZE,
            "track_box": TRACK_BOX,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })


cloud_source = None


class _QuietCloudLog(__import__("logging").Filter):
    """The page polls /cloud many times a second: keep those lines out of the server's log."""
    def filter(self, record):
        return "GET /cloud" not in record.getMessage()


__import__("logging").getLogger("werkzeug").addFilter(_QuietCloudLog())


def get_cloud():
    """The depth camera, or a recording standing in for it (CLOUD_REPLAY=path.npz, for testing without the camera)."""
    global cloud_source
    if cloud_source is None:
        replay = os.environ.get("CLOUD_REPLAY")
        if replay:
            from depth.cloud import RecordedCloud
            cloud_source = RecordedCloud(replay)
        else:
            from depth.cloud import FemtoCloud
            cloud_source = FemtoCloud()
    return cloud_source


live_fit = None


def get_live_fit():
    """The low-latency path (depth/tracker.py): fits the newest model mesh to every depth frame."""
    global live_fit
    if live_fit is None:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        from bind_mhr_joints import MHR
        from depth.fit import BodyFit
        from depth.tracker import LiveFit
        mhr = MHR(os.environ.get("MHR_MODEL", "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"))
        args = (np.asarray(faces), list(mhr.m.character_torch.skeleton.joint_names), mhr.w_joint, mhr.w_vert, mhr.w)
        live_fit = LiveFit(get_cloud(), BodyFit(*args), body_args=args)
    return live_fit


@app.get("/cloud")
def cloud():
    """Latest Femto Bolt depth image: uint16 millimetres, little-endian, row by row (depth/cloud.py). 204 while there is nothing
    newer than ?after=SEQ, 503 with the reason when the camera or its SDK is not available. Capture runs only while this is
    polled. The page turns it into points with /cloud/calibration and /cloud/rays."""
    src = get_cloud()
    if request.args.get("media") and hasattr(src, "seek"):     # a replay: the page says where its video is, every time it asks
        try:
            src.seek(float(request.args["media"]))
        except ValueError:
            pass
    item = src.get()
    if cloud_source.error:
        return jsonify({"error": cloud_source.error}), 503
    try:
        after = int(request.args.get("after", "0"))
    except ValueError:
        after = 0
    if item is None or item[0] <= after:
        return Response(status=204)
    info = cloud_source.status()
    return Response(item[1], mimetype="application/octet-stream", headers={
        "X-Cloud-Seq": str(item[0]), "X-Cloud-Media": str(getattr(cloud_source, "seq_media", {}).get(item[0], "")), "X-Cloud-Fps": str(info.get("fps", "")), "X-Cloud-Link": str(info.get("connection", "")),
        "Cache-Control": "no-store"})


recording = {"left": 0, "frames": [], "path": None, "error": None, "saved": None}


def _save_recording():
    fr, c = recording["frames"], cloud_source
    has_depth = [f["depth"] is not None for f in fr]
    w, h = (c.calibration["w"], c.calibration["h"]) if c is not None and c.calibration else (0, 0)
    out = {
        "jpeg": np.array([np.frombuffer(f["jpeg"], np.uint8) for f in fr], dtype=object),
        "verts": np.stack([f["verts"] for f in fr]), "cam_t": np.stack([f["cam_t"] for f in fr]),
        "focal": np.array([f["focal"] for f in fr]), "kp2d": np.stack([f["kp2d"] for f in fr]),
        "K": np.stack([f["K"] if f["K"] is not None else np.zeros((3, 3), np.float32) for f in fr]),
        "t": np.array([f["t"] for f in fr]), "depth_seq": np.array([f["depth_seq"] for f in fr]),
        "factory": np.array([f["factory"] for f in fr]), "has_depth": np.array(has_depth),
        "anchor": np.array([json.dumps(f["anchor"]) for f in fr]), "faces": faces,
    }
    if all(f.get("joints") is not None for f in fr):
        out["joints"] = np.stack([f["joints"] for f in fr])
    if w and any(has_depth):
        out["depth"] = np.stack([np.frombuffer(f["depth"], np.uint16).reshape(h, w) if f["depth"] is not None else np.zeros((h, w), np.uint16) for f in fr])
        out["rays"] = c.rays
        out["calibration"] = np.array(json.dumps(c.calibration))
    os.makedirs(os.path.dirname(recording["path"]), exist_ok=True)
    np.savez_compressed(recording["path"], **out)
    recording.update(frames=[], saved=recording["path"])
    print(f"RECORDED {len(fr)} frames -> {recording['path']}", flush=True)


@app.get("/record")
def record():
    """?frames=N&name=NAME starts recording the next N inferred frames of whichever stream is running (picture, mesh, camera
    translation, intrinsics used, Femto Bolt depth frame) into ../recordings/NAME.npz. Without ?frames: the state."""
    n = request.args.get("frames")
    if n is not None and recording["left"] == 0:
        name = "".join(ch for ch in request.args.get("name", time.strftime("rec_%Y%m%d_%H%M%S")) if ch.isalnum() or ch in "_-")
        get_cloud().get()                                      # depth camera on
        recording.update(left=max(1, min(int(n), 600)), frames=[], error=None, saved=None,
                         path=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recordings", name + ".npz")))
    return jsonify({k: recording[k] for k in ("left", "path", "error", "saved")} | {"captured": len(recording["frames"])})


@app.get("/cloud/calibration")
def cloud_calibration():
    """Depth image size, depth-to-colour extrinsics (R row-major, t in mm) and colour intrinsics as fractions of the picture."""
    get_cloud().get()
    if cloud_source.error:
        return jsonify({"error": cloud_source.error}), 503
    if cloud_source.calibration is None:
        return Response(status=204)                           # the camera is still starting
    return jsonify(cloud_source.calibration)


@app.get("/cloud/rays")
def cloud_rays():
    """For every depth pixel, the direction (x/z, y/z) it looks along: float32 pairs, row by row."""
    if cloud_source is None or cloud_source.rays is None:
        return Response(status=204)
    return Response(cloud_source.rays.tobytes(), mimetype="application/octet-stream")


@sock.route("/stream")
def stream_socket(ws):
    global latest_frame
    global latest_frame_owner
    global frame_id
    global dropped_frames

    owner = id(ws)
    ws_lock = threading.Lock()                    # a mesh is two messages (description, then data): never interleave them
    closed = threading.Event()

    def push_fitted():
        """Sends every fitted mesh of the low-latency path to this connection as it is made."""
        last_id = 0
        while not closed.is_set():
            lf = live_fit
            if lf is None:
                time.sleep(0.05)
                continue
            r = lf.wait(last_id, timeout=0.25)
            if r is None or r["owner"] != owner:
                continue
            last_id = r["id"]
            payload = r["verts_m"].tobytes()
            nj = 0
            if r["joints"] is not None:
                P, Rj = r["joints"]
                nj = len(P)
                payload += (P / 1000.0).astype("<f4").tobytes() + (Rj if Rj is not None else np.tile(np.eye(3), (nj, 1, 1))).astype("<f4").tobytes()
            nu = 0
            if r.get("surface_u") is not None:
                nu = len(r["surface_u"]); payload += r["surface_u"].tobytes()
            now = time.time()
            meta = {"surface": nu, "capture_timestamp_ms": r["extra"].get("client_timestamp_ms"),
                    "fitted": True, "mesh_id": r["mesh_id"], "fit_id": r["id"], "pred_cam_t": [float(x) for x in r["cam_t_m"]],
                    "joints": nj, "depth": r["info"], "focal_length": r["extra"].get("focal_length"),
                    "depth_age_ms": (now - r["t_depth"]) * 1000.0, "model_age_ms": (now - r["model_t"]) * 1000.0,
                    "sent_ms": now * 1000.0,
                    "depth_media": getattr(cloud_source, "seq_media", {}).get(r.get("depth_seq"))}   # a replay: the depth frame's own time
            try:
                with ws_lock:
                    ws.send(json.dumps(meta))
                    ws.send(payload)
            except Exception:
                return

    threading.Thread(target=push_fitted, name="push-fitted", daemon=True).start()

    while True:
        try:
            metadata_text = ws.receive()
        except Exception:
            metadata_text = None

        if metadata_text is None:
            closed.set()
            with state_lock:
                calibrations.pop(owner, None)     # the stream is gone, and so is its camera
            break

        try:
            metadata = json.loads(metadata_text)

            if metadata.get("type") == "config":
                action = metadata.get("action")

                with state_lock:
                    calibration = calibration_for(owner)
                    if action == "set_inference_mode":
                        value = metadata.get("value")
                        if value not in ("body", "full"):
                            raise ValueError(
                                "inference mode must be body or full"
                            )
                        runtime_config["inference_mode"] = value

                    elif action == "set_intrinsics_mode":
                        value = metadata.get("value")
                        if value not in ("dynamic", "fixed"):
                            raise ValueError(
                                "intrinsics mode must be dynamic or fixed"
                            )
                        calibration["mode"] = value
                        calibration["fixed_when_ready"] = False

                    elif action == "set_depth":
                        calibration["depth_fusion"] = bool(metadata.get("fusion"))
                        bolt = bool(metadata.get("bolt"))
                        if not bolt and calibration.get("factory"):
                            calibration["cam"] = None; calibration["history"].clear(); calibration["factory"] = False
                        calibration["bolt"] = bolt
                        if not calibration["depth_fusion"]:
                            calibration.pop("anchor", None)
                        if bolt:
                            get_cloud().get()                    # start the camera, so that its intrinsics are known

                    elif action == "recalibrate":
                        calibration["history"].clear()
                        calibration["cam"] = None
                        calibration["mode"] = "dynamic"
                        calibration["fixed_when_ready"] = True
                        calibration["frames"] = int(
                            metadata.get("frames", 30)
                        )

                    else:
                        raise ValueError(
                            f"Unknown configuration action: {action}"
                        )

                    response_config = {
                        "type": "config",
                        **runtime_config,
                        "intrinsics_mode": calibration["mode"],
                        "calibration_frames": calibration["frames"],
                        "calibrated": calibration["cam"] is not None,
                        "focal_length": (
                            float(calibration["cam"][0, 0, 0])
                            if calibration["cam"] is not None
                            else None
                        ),
                        "calibration_samples": len(
                            calibration["history"]
                        ),
                    }

                with ws_lock:
                    ws.send(json.dumps(response_config))
                continue

            jpeg_bytes = ws.receive()

            if not isinstance(jpeg_bytes, bytes) or len(jpeg_bytes) < 100:
                with ws_lock:
                    ws.send(json.dumps({"error": "Invalid JPEG frame"}))
                continue

            server_receive_ns = time.perf_counter_ns()
            client_timestamp_ms = str(
                metadata.get("capture_timestamp_ms", "")
            )
            encode_ms = str(metadata.get("encode_ms", 0))
            if metadata.get("media_time") is not None and hasattr(cloud_source, "seek"):
                cloud_source.seek(metadata["media_time"])        # a video file with a recorded depth stream: keep them in step

            with state_lock:
                frame_id += 1
                accepted_frame_id = frame_id

                if latest_frame is not None:
                    dropped_frames += 1

                latest_frame = (
                    accepted_frame_id,
                    client_timestamp_ms,
                    encode_ms,
                    server_receive_ns,
                    0.0,
                    jpeg_bytes,
                )
                latest_frame_owner = owner

            frame_event.set()

            with mesh_condition:
                completed = mesh_condition.wait_for(
                    lambda: (
                        latest_mesh is not None
                        and latest_mesh["source_frame_id"]
                        == accepted_frame_id
                    ),
                    timeout=30.0,
                )

                if not completed:
                    with ws_lock:
                        ws.send(json.dumps({
                            "error": "Inference timeout",
                            "source_frame_id": accepted_frame_id,
                        }))
                    continue

                current = latest_mesh
                inference_seconds = last_inference_seconds

            if current.get("no_person", False):
                with ws_lock:
                  ws.send(json.dumps({
                    "no_person": True,
                    "mesh_id": current["mesh_id"],
                    "source_frame_id": current["source_frame_id"],
                    "encode_ms": current["encode_ms"],
                    "queue_wait_ms": current["queue_wait_ms"],
                    "decode_ms": current["decode_ms"],
                    "inference_ms": inference_seconds * 1000,
                    "capture_timestamp_ms":
                        current["client_timestamp_ms"],
                }))
                continue

            # while the low-latency path is sending fitted meshes for this stream, the model's own mesh is not sent
            skip_mesh = live_fit is not None and live_fit.active(owner)
            reply = json.dumps({
                "no_person": False,
                "skip_mesh": skip_mesh,
                "mesh_id": current["mesh_id"],
                "source_frame_id": current["source_frame_id"],
                "encode_ms": current["encode_ms"],
                "queue_wait_ms": current["queue_wait_ms"],
                "decode_ms": current["decode_ms"],
                "inference_ms": inference_seconds * 1000,
                "capture_timestamp_ms": current["client_timestamp_ms"],
                "pred_keypoints_2d": current["pred_keypoints_2d"].tolist(),
                "pred_cam_t": current["pred_cam_t"].tolist(),
                "focal_length": current["focal_length"],
                "inference_mode": runtime_config["inference_mode"],
                "intrinsics_mode": current.get("intrinsics_mode", "fixed"),
                "calibrated": current.get("calibrated", False),
                "calibration_samples": current.get("calibration_samples", 0),
                "depth": current.get("depth"),
                # number of MHR joints appended to the binary message after the vertices
                # (n * 3 positions, then n * 9 rotation-matrix entries, float32); 0 = none
                "joints": (
                    len(current["joints"]) // 12
                    if current.get("joints") is not None else 0
                ),
            })

            payload = current["vertices"].astype("<f4", copy=False).tobytes()
            if current.get("joints") is not None:
                payload += current["joints"].tobytes()
            with ws_lock:
                ws.send(reply)
                if not skip_mesh:
                    ws.send(payload)

        except Exception as exc:
            try:
                with ws_lock:
                    ws.send(json.dumps({
                        "error": f"{type(exc).__name__}: {exc}"
                    }))
            except Exception:
                closed.set()
                break

def _setup_depth(spec):
    """--depth femto | recorded:<file.npz>"""
    global depth_source, depth_fusion
    from depth.source import FemtoBoltSource, RecordedSource
    from depth.fusion import DepthFusion
    depth_source = FemtoBoltSource() if spec == "femto" else RecordedSource(spec.split(":", 1)[1])
    depth_fusion_args["K"] = depth_source.K


depth_fusion_args = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=None, help="femto | recorded:<file.npz>")
    parser.add_argument("--cloth-mm", type=float, default=0.0, help="gown/blanket thickness for depth ICP")
    cli = parser.parse_args()
    if cli.depth:
        _setup_depth(cli.depth)
        depth_fusion_args["cloth_mm"] = cli.cloth_mm
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8097")),
        threaded=True,
        debug=False,
        use_reloader=False,
    )

