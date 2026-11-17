"""Measure the /stream round trip of a running webcam_server.py without a browser: send the same JPEG over the WebSocket the way
webcam.html does and time each frame from "JPEG ready" to "mesh received".
usage: python tools/stream_latency.py image.jpg [--frames 200] [--url ws://127.0.0.1:8097/stream]"""
import argparse, json, time, urllib.request
import cv2, numpy as np
from simple_websocket import Client

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--url", default="ws://127.0.0.1:8097/stream")
ap.add_argument("--size", default="640x480")
a = ap.parse_args()

http = a.url.replace("ws://", "http://").rsplit("/", 1)[0]
while True:
    try:
        if json.load(urllib.request.urlopen(http + "/status", timeout=2)).get("estimator_ready"):
            break
    except Exception:
        pass
    time.sleep(1)
print("server", json.load(urllib.request.urlopen(http + "/status")))

w, h = map(int, a.size.split("x"))
img = cv2.resize(cv2.imread(a.image), (w, h))
jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
ws = Client.connect(a.url)
rt, inf, people, calibrated_at = [], [], 0, None
for i in range(a.frames):
    t = time.perf_counter()
    ws.send(json.dumps({"capture_timestamp_ms": time.time() * 1000, "encode_ms": 0}))
    ws.send(jpeg)
    meta = json.loads(ws.receive())
    if meta.get("error"):
        print("server error:", meta["error"]); continue
    if not meta.get("no_person"):
        verts = ws.receive(); people += 1
        assert len(verts) == 18439 * 12 + int(meta.get("joints", 0)) * 48, len(verts)
        if meta.get("calibrated") and calibrated_at is None:
            calibrated_at = i
    rt.append((time.perf_counter() - t) * 1000); inf.append(meta.get("inference_ms", 0))
ws.close()
s = max(calibrated_at or 0, 20) + 10
r, f = np.array(rt[s:]), np.array(inf[s:])
print(f"frames {a.frames}, person found in {people}, focal calibrated at frame {calibrated_at}")
print(f"after warm-up ({len(r)} frames): inference median {np.median(f):.1f} ms p90 {np.percentile(f, 90):.1f} | "
      f"round trip (send JPEG -> mesh back) median {np.median(r):.1f} ms p90 {np.percentile(r, 90):.1f} max {r.max():.1f} | {1000 / np.mean(r):.1f} fps")
