"""Drive a running server with the pictures of a recording, as the page would, with depth fitting switched on, and report what
comes back: how often fitted meshes arrive, how old their depth frame is when they are sent, how long a fit takes, and how well
they sit on the depth.

Start the server with the same recording standing in for the depth camera:
    set CLOUD_REPLAY=..\\recordings\\NAME.npz   (and PORT=8098 to keep it apart from a live one), then run_local.cmd

usage: python tools/replay_stream.py ../recordings/NAME.npz [--url ws://127.0.0.1:8098/stream] [--seconds 20] [--no-fit]
"""
import json, sys, threading, time

import numpy as np
from simple_websocket import Client


def main():
    path = sys.argv[1]
    url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "ws://127.0.0.1:8098/stream"
    seconds = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 20.0
    fit = "--no-fit" not in sys.argv
    z = np.load(path, allow_pickle=True)
    jpegs = [bytes(j.tobytes()) for j in z["jpeg"]]
    ws = Client.connect(url)
    ws.send(json.dumps({"type": "config", "action": "set_depth", "fusion": fit, "bolt": True}))
    model, fitted, state = [], [], {"i": 0, "sent": 0.0, "pending_fit": None, "stop": False}

    def send_next():
        state["sent"] = time.time()
        ws.send(json.dumps({"capture_timestamp_ms": time.time() * 1000, "encode_ms": 0}))
        ws.send(jpegs[state["i"] % len(jpegs)]); state["i"] += 1

    t0 = time.time()
    first = True
    while time.time() - t0 < seconds:
        msg = ws.receive(timeout=40)
        if msg is None:
            break
        if isinstance(msg, (bytes, bytearray)):
            if state["pending_fit"] is not None:
                m = state["pending_fit"]; state["pending_fit"] = None
                n_expected = 18439 * 12 + int(m.get("joints", 0)) * 48
                fitted.append({"t": time.time(), "age": m["depth_age_ms"] + (time.time() * 1000 - m["sent_ms"]), "fit_ms": m["depth"].get("fit_ms"),
                               "median": m["depth"].get("median_mm"), "within2": m["depth"].get("within_2mm"), "model_age": m["model_age_ms"],
                               "bytes_ok": len(msg) == n_expected, "scale": m["depth"].get("scale"), "tracked": m["depth"].get("tracked"),
                               "core": m["depth"].get("core_ms"), "reused": m["depth"].get("reused")})
            continue
        m = json.loads(msg)
        if m.get("type") == "config":
            if first:
                first = False; send_next()
            continue
        if m.get("fitted"):
            state["pending_fit"] = m
            continue
        if m.get("error"):
            print("server error:", m["error"]); send_next(); continue
        model.append({"t": time.time(), "round_trip": (time.time() - state["sent"]) * 1000, "skip": bool(m.get("skip_mesh")),
                      "no_person": bool(m.get("no_person")), "inference": m.get("inference_ms"), "depth": m.get("depth")})
        send_next()
    ws.close()
    el = time.time() - t0
    mr = [x for x in model if not x["no_person"]]
    print(f"{el:.0f} s: {len(model)} model results ({len(model) / el:.1f} a second, inference median {np.median([x['inference'] for x in mr]):.0f} ms, "
          f"round trip {np.median([x['round_trip'] for x in mr]):.0f} ms), mesh left out of {sum(x['skip'] for x in mr)} replies")
    if mr and mr[-1]["depth"]:
        print("   last model reply's depth note:", mr[-1]["depth"])
    if not fitted:
        print("   no fitted meshes arrived")
        return
    f = fitted[len(fitted) // 5:]                                 # after the start-up
    gaps = np.diff([x["t"] for x in f]) * 1000
    print(f"   fitted meshes: {len(fitted)} ({len(f) / (f[-1]['t'] - f[0]['t']):.1f} a second; gap median {np.median(gaps):.0f} ms, 95th pct {np.percentile(gaps, 95):.0f} ms)")
    print(f"   fit time {np.median([x['fit_ms'] for x in f]):.0f} ms (95th pct {np.percentile([x['fit_ms'] for x in f], 95):.0f}); depth frame age when the mesh "
          f"reaches the client: median {np.median([x['age'] for x in f]):.0f} ms, 95th pct {np.percentile([x['age'] for x in f], 95):.0f} ms")
    core = [x["core"] for x in f if x["core"] is not None]
    if core:
        print(f"   of the fit time, the fit itself (in its own process): median {np.median(core):.0f} ms; measurements reused in {100 * np.mean([bool(x['reused']) for x in f]):.0f}% of fits")
    print(f"   model mesh age behind each fit: median {np.median([x['model_age'] for x in f]):.0f} ms; tracked {100 * np.mean([bool(x['tracked']) for x in f]):.0f}% of fits")
    print(f"   fit on the depth: median error {np.median([x['median'] for x in f if x['median'] is not None]):.2f} mm, "
          f"{100 * np.median([x['within2'] for x in f if x['within2'] is not None]):.0f}% of visible vertices within 2 mm; payload sizes ok: {all(x['bytes_ok'] for x in f)}")


if __name__ == "__main__":
    main()
