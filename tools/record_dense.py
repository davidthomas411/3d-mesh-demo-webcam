"""Record a dense sequence from a running server: every Femto Bolt depth frame (30 a second) and every mesh the model makes,
on one clock. The server's own recorder (GET /record) keeps the picture, mesh and camera translation of each inferred frame with
its completion time; this script polls GET /cloud alongside it and keeps every depth frame with its arrival time.

Writes NAME.npz (the server's recording) and NAME_depth.npz: depth_z (zlib-compressed uint16 frames), seq, t (seconds, the same
clock as the server's `t`), rays, calibration.

usage: python tools/record_dense.py NAME [mesh_frames=300] [--url http://127.0.0.1:8097]
"""
import json, os, sys, time, urllib.request, zlib

import numpy as np


def main():
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 300
    url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8097"
    get = lambda path: urllib.request.urlopen(url + path, timeout=5)
    state = json.load(get(f"/record?frames={n}&name={name}"))
    if state.get("left", 0) == 0:
        raise SystemExit(f"the server did not start recording: {state}")
    frames, seqs, times, seq, t0 = [], [], [], 0, time.time()
    while True:
        try:
            r = get(f"/cloud?after={seq}")
            if r.status == 200:
                data = r.read(); seq = int(r.headers["X-Cloud-Seq"])
                frames.append(zlib.compress(data, 1)); seqs.append(seq); times.append(time.time())
            else:
                time.sleep(0.004)
        except Exception:
            time.sleep(0.02)
        if len(frames) % 30 == 0 or time.time() - t0 > 600:
            st = json.load(get("/record"))
            if st.get("saved") or st.get("error") or time.time() - t0 > 600:
                break
    cal = json.load(get("/cloud/calibration"))
    rays = np.frombuffer(get("/cloud/rays").read(), np.float32).reshape(cal["h"], cal["w"], 2)
    out = os.path.join(os.path.dirname(st["path"]), name + "_depth.npz")
    np.savez(out, depth_z=np.array(frames, dtype=object), seq=np.array(seqs), t=np.array(times), rays=rays, calibration=np.array(json.dumps(cal)))
    dt = np.diff(times)
    print(f"{len(frames)} depth frames in {times[-1] - times[0]:.1f} s ({1 / np.median(dt):.1f} a second, longest gap {dt.max() * 1000:.0f} ms) -> {out}")
    print("server:", st)


if __name__ == "__main__":
    main()
