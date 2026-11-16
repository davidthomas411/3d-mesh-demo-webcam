"""Stand-in for webcam_server.py that needs no GPU and no estimator: it speaks the same /stream WebSocket protocol but answers every camera frame with the next mesh of a
recorded MHR motion clip (tools/mhr_clip.py) after a simulated inference delay. Use it to test webcam.html, anatomy/stream.js and the rig end to end, and to measure the
rig's real-time behaviour, without the Fast SAM 3D Body checkpoints.
usage: python tools/replay_server.py [--clip clips/synthetic_demo_clip.npz] [--inference-ms 90] [--port 8000] [--mhr-onnx PATH --mhr-rest PATH]
Protocol (as webcam_server.py): client sends JSON {capture_timestamp_ms, encode_ms} then a JPEG; server replies JSON metadata then 18439 x 3 float32 vertices."""
import argparse, json, os, sys, time
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
from flask_sock import Sock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mhr_clip import load_clip, prepare

app = Flask(__name__); sock = Sock(app)
CFG = {}


class Player:
    def __init__(self, clip, onnx, rest):
        import onnxruntime as ort
        from scipy.interpolate import CubicSpline
        arr, fps, _ = load_clip(clip); P = prepare(arr)
        t = np.arange(len(P)) / fps
        self.spline = CubicSpline(t, P, axis=0); self.dur = t[-1]
        self.sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]); self.faces = np.load(rest)["faces"].astype(np.int32).reshape(-1)
        self.t0 = time.perf_counter()

    def mesh(self):
        t = (time.perf_counter() - self.t0) % self.dur                                  # loops
        p = self.spline(t).astype(np.float32)[None]
        return self.sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype("<f4")


@app.get("/")
def index():
    return send_from_directory(ROOT, "webcam.html")


@app.get("/anatomy/<name>")
def anatomy_js(name):
    if name not in ("anatomy.js", "stream.js", "xray.js"):
        return Response("Not found", status=404)
    return send_from_directory(os.path.join(ROOT, "anatomy"), name, mimetype="text/javascript")


@app.get("/atlas/<name>")
def atlas_file(name):
    if name not in ("atlas.json", "atlas.bin"):
        return Response("Not found", status=404)
    return send_from_directory(os.path.join(ROOT, "atlas"), name)


@app.get("/topology")
def topology():
    return jsonify({"ready": True, "vertex_count": 18439, "faces": CFG["player"].faces.tolist()})


@app.get("/status")
def status():
    return jsonify({"ready": True, "replay": True})


@sock.route("/stream")
def stream(ws):
    n = 0
    while True:
        meta_text = ws.receive()
        if meta_text is None:
            break
        meta = json.loads(meta_text)
        if meta.get("type") == "config":
            ws.send(json.dumps({"type": "config", "inference_mode": "body", "intrinsics_mode": "fixed", "calibrated": True, "focal_length": 1000.0}))
            continue
        jpeg = ws.receive()
        t0 = time.perf_counter()
        time.sleep(CFG["inference_ms"] / 1000.0)                                         # the estimator's latency
        v = CFG["player"].mesh(); n += 1
        ws.send(json.dumps({
            "no_person": False, "mesh_id": n, "source_frame_id": n, "encode_ms": float(meta.get("encode_ms", 0)), "queue_wait_ms": 0.0, "decode_ms": 1.0,
            "inference_ms": (time.perf_counter() - t0) * 1000, "capture_timestamp_ms": meta.get("capture_timestamp_ms"),
            "pred_keypoints_2d": np.zeros((70, 2)).tolist(), "pred_cam_t": [0.0, 0.0, 3.0], "focal_length": 1000.0,
            "inference_mode": "body", "intrinsics_mode": "fixed", "calibrated": True, "calibration_samples": 0, "depth": None}))
        ws.send(v.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(ROOT, "clips", "synthetic_demo_clip.npz"))
    ap.add_argument("--inference-ms", type=float, default=90.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mhr-onnx", default=os.environ.get("ONNX")); ap.add_argument("--mhr-rest", default=os.environ.get("REST"))
    a = ap.parse_args()
    CFG["inference_ms"] = a.inference_ms; CFG["player"] = Player(a.clip, a.mhr_onnx, a.mhr_rest)
    app.run(host="0.0.0.0", port=a.port, threaded=True)
