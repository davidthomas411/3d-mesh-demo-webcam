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


@sock.route("/stream")
def stream_socket(ws):
    global latest_frame
    global latest_frame_owner
    global frame_id
    global dropped_frames

    owner = id(ws)

    while True:
        try:
            metadata_text = ws.receive()
        except Exception:
            metadata_text = None

        if metadata_text is None:
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

                ws.send(json.dumps(response_config))
                continue

            jpeg_bytes = ws.receive()

            if not isinstance(jpeg_bytes, bytes) or len(jpeg_bytes) < 100:
                ws.send(json.dumps({"error": "Invalid JPEG frame"}))
                continue

            server_receive_ns = time.perf_counter_ns()
            client_timestamp_ms = str(
                metadata.get("capture_timestamp_ms", "")
            )
            encode_ms = str(metadata.get("encode_ms", 0))

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
                    ws.send(json.dumps({
                        "error": "Inference timeout",
                        "source_frame_id": accepted_frame_id,
                    }))
                    continue

                current = latest_mesh
                inference_seconds = last_inference_seconds

            if current.get("no_person", False):
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

            ws.send(json.dumps({
                "no_person": False,
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
            }))

            payload = current["vertices"].astype("<f4", copy=False).tobytes()
            if current.get("joints") is not None:
                payload += current["joints"].tobytes()
            ws.send(payload)

        except Exception as exc:
            ws.send(json.dumps({
                "error": f"{type(exc).__name__}: {exc}"
            }))

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

