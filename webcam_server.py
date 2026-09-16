import json
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_sock import Sock

from mocap.core.setup_estimator import build_default_estimator

app = Flask(__name__)
sock = Sock(app)

state_lock = threading.Lock()
frame_event = threading.Event()
mesh_condition = threading.Condition(state_lock)

latest_frame = None
latest_mesh = None
frame_id = 0
mesh_id = 0
dropped_frames = 0
last_inference_seconds = None
last_error = None
estimator_ready = False
faces = None
focal_length_history = []
cached_cam_intrinsics = None
FOCAL_CALIBRATION_FRAMES = 30

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
    global focal_length_history
    global cached_cam_intrinsics
    global runtime_config

    print("Loading persistent Fast SAM 3D Body estimator...", flush=True)

    estimator = build_default_estimator(
        image_size=512,
        yolo_model_path="checkpoints/yolo/yolo11m-pose.pt",
        fov_model_size="s",
        fov_resolution_level=0,
        fov_fixed_size=512,
        fov_fast_mode=True,
    )

    faces = np.asarray(estimator.faces, dtype=np.int32)
    estimator_ready = True

    print(
        f"ESTIMATOR_READY vertices=18439 faces={len(faces)}",
        flush=True,
    )

    while True:
        frame_event.wait()
        frame_event.clear()

        with state_lock:
            item = latest_frame
            latest_frame = None

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
                intrinsics_mode = runtime_config["intrinsics_mode"]

            active_cam_intrinsics = (
                cached_cam_intrinsics
                if intrinsics_mode == "fixed"
                else None
            )

            outputs = estimator.process_one_image(
                image_rgb,
                cam_int=active_cam_intrinsics,
                inference_type=inference_mode,
                hand_box_source="yolo_pose",
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if len(outputs) != 1:
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

            print("KEYPOINT_SHAPE", np.asarray(outputs[0]["pred_keypoints_2d"]).shape, flush=True)
            focal_length = float(
                np.asarray(outputs[0]["focal_length"]).reshape(-1)[0]
            )
            focal_length_history.append(focal_length)
            if len(focal_length_history) > 300:
                del focal_length_history[:-300]

            if (
                cached_cam_intrinsics is None
                and len(focal_length_history) >= runtime_config["calibration_frames"]
            ):
                calibrated_focal = float(
                    np.median(
                        focal_length_history[
                            :runtime_config["calibration_frames"]
                        ]
                    )
                )
                height, width = image_rgb.shape[:2]
                cached_cam_intrinsics = torch.tensor(
                    [[
                        [calibrated_focal, 0.0, width / 2.0],
                        [0.0, calibrated_focal, height / 2.0],
                        [0.0, 0.0, 1.0],
                    ]],
                    dtype=torch.float32,
                )
                print(
                    f"FOCAL_CALIBRATED frames={FOCAL_CALIBRATION_FRAMES} "
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

            with state_lock:
                mesh_id += 1
                latest_mesh = {
                    "mesh_id": mesh_id,
                    "source_frame_id": current_frame_id,
                    "no_person": False,
                    "vertices": vertices,
                    "server_receive_wall_ms": server_receive_wall_ms,
                    "encode_ms": float(encode_ms),
                    "queue_wait_ms": queue_wait_ms,
                    "decode_ms": decode_ms,
                    "pred_keypoints_2d": to_numpy(outputs[0]["pred_keypoints_2d"]),
                    "pred_cam_t": to_numpy(outputs[0]["pred_cam_t"]).reshape(-1),
                    "focal_length": focal_length,
                    "client_timestamp_ms": client_timestamp_ms,
                }
                last_inference_seconds = elapsed
                last_error = None
                mesh_condition.notify_all()

            print(
                f"MESH {mesh_id} frame={current_frame_id} "
                f"inference={elapsed:.3f}s",
                flush=True,
            )

        except Exception as exc:
            with state_lock:
                last_error = f"{type(exc).__name__}: {exc}"
            print(f"INFERENCE_ERROR: {last_error}", flush=True)


@app.get("/")
def index():
    return send_from_directory(".", "webcam.html")


@app.post("/frame")
def receive_frame():
    global latest_frame
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

    response = Response(
        current["vertices"].astype("<f4", copy=False).tobytes(),
        content_type="application/octet-stream",
    )
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
    values = np.asarray(focal_length_history, dtype=np.float64)

    if values.size == 0:
        return jsonify({"count": 0})

    median = float(np.median(values))

    return jsonify({
        "count": int(values.size),
        "calibrated": cached_cam_intrinsics is not None,
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
        })


@sock.route("/stream")
def stream_socket(ws):
    global cached_cam_intrinsics
    global latest_frame
    global frame_id
    global dropped_frames

    while True:
        metadata_text = ws.receive()

        if metadata_text is None:
            break

        try:
            metadata = json.loads(metadata_text)

            if metadata.get("type") == "config":
                action = metadata.get("action")

                with state_lock:
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
                        runtime_config["intrinsics_mode"] = value

                    elif action == "recalibrate":
                        focal_length_history.clear()
                        cached_cam_intrinsics = None
                        runtime_config["intrinsics_mode"] = "dynamic"
                        runtime_config["calibration_frames"] = int(
                            metadata.get("frames", 30)
                        )

                    else:
                        raise ValueError(
                            f"Unknown configuration action: {action}"
                        )

                    response_config = {
                        "type": "config",
                        **runtime_config,
                        "calibrated": cached_cam_intrinsics is not None,
                        "focal_length": (
                            float(cached_cam_intrinsics[0, 0, 0])
                            if cached_cam_intrinsics is not None
                            else None
                        ),
                        "calibration_samples": len(
                            focal_length_history
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
                "intrinsics_mode": runtime_config["intrinsics_mode"],
                "calibrated": cached_cam_intrinsics is not None,
                "calibration_samples": len(focal_length_history),
            }))

            ws.send(
                current["vertices"]
                .astype("<f4", copy=False)
                .tobytes()
            )

        except Exception as exc:
            ws.send(json.dumps({
                "error": f"{type(exc).__name__}: {exc}"
            }))

if __name__ == "__main__":
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    app.run(
        host="0.0.0.0",
        port=8097,
        threaded=True,
        debug=False,
        use_reloader=False,
    )

