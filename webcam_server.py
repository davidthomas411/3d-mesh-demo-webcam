import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from mocap.core.setup_estimator import build_default_estimator

app = Flask(__name__)

state_lock = threading.Lock()
frame_event = threading.Event()

latest_frame = None
latest_mesh = None
frame_id = 0
mesh_id = 0
dropped_frames = 0
last_inference_seconds = None
last_error = None
estimator_ready = False
faces = None


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

        current_frame_id, client_timestamp_ms, jpeg_bytes = item

        image = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            with state_lock:
                last_error = "JPEG decode failed"
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()

        try:
            outputs = estimator.process_one_image(
                image_rgb,
                cam_int=None,
                hand_box_source="yolo_pose",
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if len(outputs) != 1:
                with state_lock:
                    last_inference_seconds = elapsed
                    last_error = (
                        f"Expected one person, detected {len(outputs)}"
                    )
                continue

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
                    "vertices": vertices,
                    "pred_cam_t": to_numpy(outputs[0]["pred_cam_t"]).reshape(-1),
                    "focal_length": float(np.asarray(outputs[0]["focal_length"]).reshape(-1)[0]),
                    "client_timestamp_ms": client_timestamp_ms,
                }
                last_inference_seconds = elapsed
                last_error = None

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

    jpeg_bytes = request.get_data(cache=False)
    client_timestamp_ms = request.headers.get("X-Capture-Timestamp-Ms", "")

    if len(jpeg_bytes) < 100:
        return Response("Invalid frame", status=400)

    with state_lock:
        frame_id += 1
        if latest_frame is not None:
            dropped_frames += 1
        latest_frame = (frame_id, client_timestamp_ms, jpeg_bytes)
        accepted_frame_id = frame_id

    frame_event.set()

    response = Response(status=204)
    response.headers["X-Frame-Id"] = str(accepted_frame_id)
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
    response.headers["X-Capture-Timestamp-Ms"] = current["client_timestamp_ms"]
    response.headers["X-Pred-Cam-T"] = ",".join(map(str, current["pred_cam_t"]))
    response.headers["X-Focal-Length"] = str(current["focal_length"])
    return response


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
