import json
import threading
import time
from pathlib import Path

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
anatomical_regions = None
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



class HeadShouldersDetector:
    """
    Persistent DEIMv2 whole-head detector.

    DEIMv2 class 7 provides the native head box. That box is
    expanded downward and laterally to create the box supplied
    to SAM 3D Body.
    """

    def __init__(
        self,
        model_path=(
            "checkpoints/insightface/"
            "deimv2_dinov3_s_wholebody49_"
            "ins_s08_maskhead256x3_center_"
            "1240query_masks.onnx"
        ),
        score_threshold=0.25,
        smoothing_alpha=0.35,
        max_missing_frames=4,
    ):
        from pathlib import Path

        from tools.hrffa_reference import (
            FaceAligner,
            HeadDetector,
            build_providers,
        )

        providers = build_providers("cuda")

        self.detector = HeadDetector(
            model_path=Path(model_path),
            providers=providers,
            score_threshold=float(score_threshold),
        )

        self.aligner = FaceAligner(
            model_path=Path(
                "checkpoints/insightface/"
                "hrffa_hg0_ibug68_1x3x96x96.onnx"
            ),
            providers=providers,
            input_norm="center05",
            crop_pad=0.05,
        )

        self.smoothing_alpha = float(smoothing_alpha)
        self.max_missing_frames = int(max_missing_frames)

        self.filtered_box = None
        self.missing_frames = 0

        self.last_head_box = None
        self.last_box = None
        self.last_landmarks = None
        self.last_visibility = None
        self.last_score = None
        self.last_alignment_ms = 0.0
        self.last_visibility = None
        self.last_score = None
        self.last_detection_ms = 0.0
        self.last_alignment_ms = 0.0

    def reset(self):
        self.filtered_box = None
        self.missing_frames = 0
        self.last_head_box = None
        self.last_box = None
        self.last_landmarks = None

    def warmup(self, height=480, width=640, iterations=3):
        image_bgr = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        for _ in range(iterations):
            heads = self.detector(image_bgr)

        self.last_detection_ms = (
            self.detector.last_inference_time * 1000.0
        )

        # The blank frame normally produces no head, so use a
        # centered synthetic HeadBox only to initialize HRFFA.
        from tools.hrffa_reference import HeadBox

        warm_head = HeadBox(
            width * 0.35,
            height * 0.20,
            width * 0.65,
            height * 0.65,
            1.0,
        )

        self.aligner(image_bgr, [warm_head])

        self.last_alignment_ms = (
            self.aligner.last_inference_time * 1000.0
        )

    def _smooth_box(self, measured_box):
        measured_box = np.asarray(
            measured_box,
            dtype=np.float32,
        )

        if self.filtered_box is None:
            self.filtered_box = measured_box
        else:
            alpha = self.smoothing_alpha

            self.filtered_box = (
                alpha * measured_box
                + (1.0 - alpha) * self.filtered_box
            )

        return self.filtered_box.copy()

    @staticmethod
    def _clip_box(box, width, height):
        box = np.asarray(
            box,
            dtype=np.float32,
        ).copy()

        box[0] = np.clip(box[0], 0, width - 2)
        box[1] = np.clip(box[1], 0, height - 2)

        box[2] = np.clip(
            box[2],
            box[0] + 1,
            width - 1,
        )

        box[3] = np.clip(
            box[3],
            box[1] + 1,
            height - 1,
        )

        return box

    def detect(self, image_rgb):
        height, width = image_rgb.shape[:2]

        image_bgr = cv2.cvtColor(
            np.ascontiguousarray(image_rgb),
            cv2.COLOR_RGB2BGR,
        )

        heads = self.detector(image_bgr)

        self.last_detection_ms = (
            self.detector.last_inference_time * 1000.0
        )

        if not heads:
            self.missing_frames += 1
            self.last_head_box = None
            self.last_box = None
            self.last_landmarks = None
            self.last_visibility = None
            self.last_score = None
            self.last_alignment_ms = 0.0

            if self.missing_frames > self.max_missing_frames:
                self.reset()

            return np.empty(
                (0, 4),
                dtype=np.float32,
            )

        self.missing_frames = 0

        # HeadDetector already sorts by descending confidence.
        head = heads[0]

        native_head_box = self._clip_box(
            [
                head.x1,
                head.y1,
                head.x2,
                head.y2,
            ],
            width,
            height,
        )

        head_width = max(
            native_head_box[2] - native_head_box[0],
            1.0,
        )

        head_height = max(
            native_head_box[3] - native_head_box[1],
            1.0,
        )

        center_x = 0.5 * (
            native_head_box[0] +
            native_head_box[2]
        )

        # Engineering starting values:
        # width = 2.2 head widths
        # top = 0.10 head heights above detected head
        # bottom = 1.25 head heights below detected head
        half_width = 1.10 * head_width

        measured_box = self._clip_box(
            [
                center_x - half_width,
                native_head_box[1] - 0.10 * head_height,
                center_x + half_width,
                native_head_box[3] + 1.25 * head_height,
            ],
            width,
            height,
        )

        filtered_box = self._clip_box(
            self._smooth_box(measured_box),
            width,
            height,
        )

        alignment_results = self.aligner(
            image_bgr,
            [head],
        )

        self.last_alignment_ms = (
            self.aligner.last_inference_time * 1000.0
        )

        if alignment_results:
            self.last_landmarks = (
                alignment_results[0]
                .points
                .astype(np.float32, copy=True)
            )

            self.last_visibility = (
                alignment_results[0]
                .visibility
                .astype(np.int64, copy=True)
            )
        else:
            self.last_landmarks = None
            self.last_visibility = None

        self.last_head_box = native_head_box
        self.last_box = filtered_box
        self.last_score = float(head.score)

        return filtered_box.reshape(1, 4)



def inference_worker():
    global latest_frame
    global latest_mesh
    global mesh_id
    global dropped_frames
    global last_inference_seconds
    global last_error
    global estimator_ready
    global faces
    global anatomical_regions
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

    print(
        "Loading MHR rig-derived anatomical regions...",
        flush=True,
    )

    mhr_model_path = (
        Path("checkpoints/sam-3d-body-dinov3")
        / "assets"
        / "mhr_model.pt"
    )

    mhr_model = torch.jit.load(
        str(mhr_model_path),
        map_location="cpu",
    )

    joint_names = list(
        mhr_model.get_joint_names()
    )

    joint_indices, joint_weights = (
        mhr_model.get_lbsw()
    )

    joint_indices = (
        joint_indices
        .to(torch.int64)
        .cpu()
    )

    joint_weights = (
        joint_weights
        .to(torch.float32)
        .cpu()
    )

    region_patterns = {
        "head": (
            "c_head",
            "c_jaw",
            "c_teeth",
            "c_tongue",
            "r_eye",
            "l_eye",
        ),
        "upperChest": (
            "c_spine3",
            "r_clavicle",
            "l_clavicle",
        ),
        "thorax": (
            "c_spine2",
            "c_spine3",
            "r_clavicle",
            "l_clavicle",
        ),
        "abdomen": (
            "c_spine0",
            "c_spine1",
        ),
        "pelvis": (
            "root",
            "l_upleg",
            "r_upleg",
        ),
        "arms": (
            "r_clavicle",
            "r_uparm",
            "r_lowarm",
            "r_wrist",
            "r_thumb",
            "l_clavicle",
            "l_uparm",
            "l_lowarm",
            "l_wrist",
            "l_thumb",
        ),
        "torso": (
            "root",
            "c_spine0",
            "c_spine1",
            "c_spine2",
            "c_spine3",
            "r_clavicle",
            "l_clavicle",
        ),
    }

    def matching_joint_ids(patterns):
        return {
            index
            for index, name in enumerate(joint_names)
            if any(
                name == pattern
                or name.startswith(pattern + "_")
                for pattern in patterns
            )
        }

    anatomical_regions = {
        "fullBody": list(
            range(joint_indices.shape[0])
        )
    }

    for region_name, patterns in (
        region_patterns.items()
    ):
        region_joint_ids = matching_joint_ids(
            patterns
        )

        membership = torch.zeros(
            joint_indices.shape[0],
            dtype=torch.float32,
        )

        for column in range(
            joint_indices.shape[1]
        ):
            selected = torch.zeros(
                joint_indices.shape[0],
                dtype=torch.bool,
            )

            for joint_id in region_joint_ids:
                selected |= (
                    joint_indices[:, column]
                    == joint_id
                )

            membership += torch.where(
                selected,
                joint_weights[:, column],
                torch.zeros_like(membership),
            )

        region_vertex_indices = (
            torch.nonzero(
                membership >= 0.25,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )

        anatomical_regions[region_name] = (
            region_vertex_indices
        )

    print(
        "MHR_ANATOMICAL_REGIONS_READY "
        + " ".join(
            f"{name}={len(indices)}"
            for name, indices
            in anatomical_regions.items()
        ),
        flush=True,
    )

    del mhr_model
    del joint_indices
    del joint_weights

    print(
        "Loading DEIMv2 whole-head detector...",
        flush=True,
    )

    head_shoulders_detector = HeadShouldersDetector()

    print(
        "Warming DEIMv2 whole-head detector...",
        flush=True,
    )

    head_shoulders_detector.warmup()

    print(
        "HEAD_TRACKING_READY "
        f"providers={head_shoulders_detector.detector.providers} "
        f"detector_ms="
        f"{head_shoulders_detector.last_detection_ms:.1f} "
        f"aligner_ms="
        f"{head_shoulders_detector.last_alignment_ms:.1f}",
        flush=True,
    )

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

            if inference_mode == "head_shoulders":
                head_boxes = (
                    head_shoulders_detector.detect(
                        image_rgb
                    )
                )

                if len(head_boxes) == 0:
                    outputs = []
                else:
                    outputs = estimator.process_one_image(
                        image_rgb,
                        bboxes=head_boxes,
                        cam_int=active_cam_intrinsics,
                        inference_type="body",
                    )
            else:
                outputs = estimator.process_one_image(
                    image_rgb,
                    cam_int=active_cam_intrinsics,
                    inference_type=inference_mode,
                    hand_box_source="yolo_pose",
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if len(outputs) > 1:
                def output_box_area(output):
                    box = np.asarray(
                        output.get("bbox", []),
                        dtype=np.float32,
                    ).reshape(-1)

                    if box.size < 4:
                        return 0.0

                    return max(
                        0.0,
                        float(box[2] - box[0]),
                    ) * max(
                        0.0,
                        float(box[3] - box[1]),
                    )

                outputs = [
                    max(
                        outputs,
                        key=output_box_area,
                    )
                ]

            if len(outputs) == 0:
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
                    "mediapipe_landmarks": None,
                    "head_landmarks": (
                        head_shoulders_detector
                        .last_landmarks
                        .copy()
                        if (
                            inference_mode == "head_shoulders"
                            and head_shoulders_detector
                            .last_landmarks
                            is not None
                        )
                        else None
                    ),
                    "head_landmark_visibility": (
                        head_shoulders_detector
                        .last_visibility
                        .copy()
                        if (
                            inference_mode == "head_shoulders"
                            and head_shoulders_detector
                            .last_visibility
                            is not None
                        )
                        else None
                    ),
                    "head_alignment_ms": (
                        float(
                            head_shoulders_detector
                            .last_alignment_ms
                        )
                        if inference_mode == "head_shoulders"
                        else 0.0
                    ),
                    "head_shoulders_box": (
                        head_shoulders_detector.last_box.copy()
                        if (
                            inference_mode == "head_shoulders"
                            and head_shoulders_detector.last_box
                            is not None
                        )
                        else None
                    ),
                    "head_detector_box": (
                        head_shoulders_detector.last_head_box.copy()
                        if (
                            inference_mode == "head_shoulders"
                            and head_shoulders_detector.last_head_box
                            is not None
                        )
                        else None
                    ),
                    "head_detection_ms": (
                        float(
                            head_shoulders_detector.last_detection_ms
                        )
                        if inference_mode == "head_shoulders"
                        else 0.0
                    ),
                    "head_detection_score": (
                        float(head_shoulders_detector.last_score)
                        if (
                            inference_mode == "head_shoulders"
                            and head_shoulders_detector.last_score
                            is not None
                        )
                        else None
                    ),
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
        "anatomical_regions": anatomical_regions,
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
                        if value not in (
                            "body",
                            "full",
                            "head_shoulders",
                        ):
                            raise ValueError(
                                "inference mode must be "
                                "body, full, or head_shoulders"
                            )

                        previous_value = runtime_config[
                            "inference_mode"
                        ]

                        runtime_config["inference_mode"] = value

                        if (
                            value == "head_shoulders"
                            or previous_value
                            == "head_shoulders"
                        ):
                            head_shoulders_detector.reset()

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
                "mediapipe_landmarks": (
                    current["mediapipe_landmarks"].tolist()
                    if current.get("mediapipe_landmarks")
                    is not None
                    else None
                ),
                "head_shoulders_box": (
                    current["head_shoulders_box"].tolist()
                    if current.get("head_shoulders_box")
                    is not None
                    else None
                ),
                "head_landmarks": (
                    current["head_landmarks"].tolist()
                    if current.get("head_landmarks")
                    is not None
                    else None
                ),
                "head_landmark_visibility": (
                    current[
                        "head_landmark_visibility"
                    ].tolist()
                    if current.get(
                        "head_landmark_visibility"
                    ) is not None
                    else None
                ),
                "head_alignment_ms": float(
                    current.get("head_alignment_ms", 0.0)
                ),
                "head_detector_box": (
                    current["head_detector_box"].tolist()
                    if current.get("head_detector_box")
                    is not None
                    else None
                ),
                "head_detection_ms": float(
                    current.get("head_detection_ms", 0.0)
                ),
                "head_detection_score": current.get(
                    "head_detection_score"
                ),
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

