"""A gantry recording made playable in the LIVE page: mesh, point cloud, skeleton and CT, exactly as with the camera.

The live page already plays a video file as its camera, and the server already replays a recorded depth stream in place of
the Femto Bolt (CLOUD_REPLAY). This writes both from a gantry recording:

  WEB_DIR/replay.mp4   the colour picture, turned head-up, frames a:b
  WEB_DIR/replay.npz   the depth of the same frames in the same (turned) picture, half size, with the calibration the
                       server expects. The recording's depth is already aligned to the colour picture, so the "depth
                       camera" here IS the colour camera: no extrinsics, no lens distortion.

The page tells the server the video's time with every picture, and the server serves the depth frame of that moment, so
the two cannot drift.

    python tools/gantry_replay.py FOLDER WEB_DIR [--frames 4380:5580:2] [--turn 180]
    run_gantry_replay.cmd            then open http://127.0.0.1:8096/?bolt=1&video=/gantry-data/replay.mp4
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gantry_recording import GantryRecording, parse_frames, turned  # noqa: E402


def main():
    import cv2
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rec = GantryRecording(sys.argv[1]); web = sys.argv[2]
    os.makedirs(web, exist_ok=True)
    frames = parse_frames(arg("--frames", "4380:5580:2"), rec.n); turn = int(arg("--turn", 180))
    fps = 30.0 / (frames[1] - frames[0])
    W, H = rec.size
    K = rec.K.copy()
    if turn == 180:
        K[0, 2], K[1, 2] = W - 1 - K[0, 2], H - 1 - K[1, 2]
    elif turn:
        raise SystemExit("only --turn 0 or 180 (the picture must stay landscape)")
    vw = None
    for fourcc, name in (("avc1", "replay.mp4"), ("VP80", "replay.webm")):
        vw = cv2.VideoWriter(os.path.join(web, name), cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if vw.isOpened():
            video = name
            break
        vw.release(); vw = None
    if vw is None:
        raise SystemExit("this OpenCV build can write neither H.264 nor VP8")
    D = np.zeros((len(frames), H // 2, W // 2), np.uint16)
    for n, i in enumerate(frames):
        vw.write(turned(rec.color(i), turn))
        D[n] = turned(rec.depth(i), turn)[::2, ::2]
    vw.release()
    h, w = D.shape[1:]
    kd = [K[0, 0] / 2, K[1, 1] / 2, K[0, 2] / 2, K[1, 2] / 2]       # the half-size depth picture's own intrinsics
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(uu - kd[2]) / kd[0], (vv - kd[3]) / kd[1]], -1).astype(np.float32)
    cal = {"w": w, "h": h, "R": np.eye(3).ravel().tolist(), "t": [0.0, 0.0, 0.0],
           "k": [K[0, 0] / W, K[1, 1] / H, K[0, 2] / W, K[1, 2] / H], "kd": kd, "dist": [0.0] * 8}
    np.savez_compressed(os.path.join(web, "replay.npz"), depth=D, has_depth=np.ones(len(frames), bool), t=np.arange(len(frames)) / fps,
                        rays=rays, calibration=np.array(json.dumps(cal)), source_frames=np.array(frames))
    print("wrote", video, "and replay.npz:", len(frames), "frames at", fps, "a second, depth", w, "x", h)


if __name__ == "__main__":
    main()
