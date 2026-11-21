"""The depth fit in a process of its own (started by depth/tracker.py: python -m depth.fitproc PORT KEY).

It needs numpy, scipy and OpenCV only: the body model's skinning weights arrive in the set-up message, so nothing here loads
the model or touches the GPU. Messages, over a multiprocessing connection:

  ("setup", (faces, joint_names, w_joint, w_vert, w), rays, calibration, refit_every)
  ("model", {owner, mesh_id, verts, jpos, jrot, cam_t_mm})     the newest model mesh, when it changes
  ("depth", bytes)                                             one depth frame -> replies with the fit result, or None
  ("reset",)                                                   forget the tracking state
  ("stop",)
"""
import sys
import time

import numpy as np


def main():
    from multiprocessing.connection import Client
    from depth.fit import BodyFit, DepthFrame
    from depth.tracker import FitState, fit_one
    conn = Client(("127.0.0.1", int(sys.argv[1])), authkey=bytes.fromhex(sys.argv[2]))
    body = rays = cal = model = None
    state, refit_every = FitState(), 45
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        kind = msg[0]
        if kind == "stop":
            break
        if kind == "setup":
            args, rays, cal, refit_every = msg[1], msg[2], msg[3], msg[4]
            body = BodyFit(*args)
        elif kind == "model":
            model = msg[1]
        elif kind == "reset":
            state = FitState()
        elif kind == "depth":
            out = None
            if body is not None and model is not None:
                try:
                    t0 = time.perf_counter()
                    depth = np.frombuffer(msg[1], np.uint16).reshape(cal["h"], cal["w"])
                    out = fit_one(state, body, model, DepthFrame(depth, rays, cal["R"], cal["t"], cal["kd"], cal["dist"]), refit_every)
                    if out is not None:
                        out["info"]["core_ms"] = round((time.perf_counter() - t0) * 1000, 1)    # the fit alone, without the hand-over
                except Exception as exc:                         # report, do not die: the next frame may be fine
                    print(f"fit process: {type(exc).__name__}: {exc}", flush=True)
                    state = FitState()
            conn.send(out)
    conn.close()


if __name__ == "__main__":
    main()
