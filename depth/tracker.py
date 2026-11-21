"""The low-latency path: every depth frame, fit the newest body mesh to it and publish the result.

Two clocks run in this demo. The RGB model produces a posed mesh from a picture that is, by the time it is shown, 150-350 ms
old, five to seven times a second. The depth camera delivers the true surface thirty times a second, a few tens of
milliseconds old. Showing the model's mesh next to the live depth makes the mesh trail behind the person.

Here the model's mesh is no longer what is shown. It is the SHAPE AND POSE PRIOR: each new depth frame is fitted with the
newest model mesh (depth/fit.py), starting from the previous frame's fit, and the fitted mesh is what goes to the page. What the
viewer sees is therefore as old as the depth frame plus one fit, and it sits on the measured surface. When the model delivers a
new mesh, the next fit simply starts from it with the previous part transforms: the mesh carries what the picture knows (pose,
limbs the depth camera cannot see), the transforms carry where the body is now.

The body is placed AS ONE (one scale, one small rotation, one translation: depth/fit.py); its pose stays the model's. A fit
that fails (the body left the depth image) starts again from scratch on the next frame; meanwhile the last placement that
fitted stands in, so the mesh does not jump back to the model's own, wrong, distance.

The fit runs in a PROCESS OF ITS OWN (depth/fitproc.py) when it can be started: in this process it shares the Python
interpreter with the model loop and a 25 ms fit takes 43 ms. The two talk over a local connection; a depth frame and a mesh
are a megabyte between them.
"""
import os
import subprocess
import sys
import threading
import time

import numpy as np

from .fit import DepthFrame


class FitState:
    """What carries over from one frame's fit to the next."""
    def __init__(self):
        self.M, self.owner, self.n, self.track = None, None, 0, {}
        self.M_good, self.held = None, 0      # the last transforms that fitted, and for how many frames they have stood in


def fit_one(state, body, model, frame, refit_every=10 ** 9, hold_frames=45):
    """Fit model["verts"] to one depth frame, tracking from `state`. -> result dict (no timing, no ids) or None."""
    if model["owner"] != state.owner or state.n >= refit_every:
        state.M, state.owner, state.n, state.track = None, model["owner"], 0, {}
    V = model["verts"]
    if model["mesh_id"] != state.track.get("mesh_id"):
        # A new model mesh: its shape differs a little from the last one (the model's own frame-to-frame noise), so the
        # surface offsets worked out for the last one do not belong to it. They start afresh on this frame and are averaged
        # over the frames that use this mesh.
        state.track["u"] = None; state.track["mesh_id"] = model["mesh_id"]
    X, Mn, info = body.fit(V, frame, init=state.M, pose_iters=1 if state.M is not None else 2, track=state.track)
    if Mn is None and state.M is not None:                       # tracking lost: once more from scratch on this frame
        state.track = {"mesh_id": model["mesh_id"]}
        X, Mn, info = body.fit(V, frame, init=None, pose_iters=2, track=state.track)
    state.M = Mn
    state.n += 1
    if Mn is None:
        # No fit on this frame (the body left the depth image, came too near, was covered). The mesh does not fall back to
        # the model's own placement, 10-20% farther away: the last placement that fitted stands in for a while, on the
        # model's newest pose, without the skin layer.
        if state.M_good is None or state.held >= hold_frames:
            state.M_good = None
            return None
        state.held += 1
        Mn, held = state.M_good, True
        X_pose, u = V @ Mn[0, :3, :3].T + Mn[0, :3, 3], None
        info = {"ok": True, "held": True, "reason": info.get("reason")}
    else:
        state.M_good, state.held, held = Mn, 0, False
        X_pose, u = state.track.get("X_pose", X), state.track.get("u")
    # the body origin the page places the mesh at: where the transform carries the model's own origin
    t_new = Mn[0, :3, :3] @ model["cam_t_mm"] + Mn[0, :3, 3]
    joints = None
    if model.get("jpos") is not None:
        P, Rj = body.move_joints(model["jpos"], model.get("jrot"), Mn)
        joints = ((P - t_new).astype(np.float32), None if Rj is None else Rj.astype(np.float32))
    info.update(fit=True, samples=info.get("surface_vertices", 0), scale=round(float(np.cbrt(np.linalg.det(Mn[0, :3, :3]))), 4),
                tracked=state.n > 1, held=held)
    # verts_m: the PLACED mesh (the model's pose, moved as one): what the skeleton is built on. surface_u: per vertex, how
    # much farther along its ray the measured skin is (a fraction of its distance); the page draws the skin with it.
    return {"verts_m": ((X_pose - t_new) / 1000.0).astype("<f4"), "cam_t_m": (t_new / 1000.0).astype(np.float32), "joints": joints,
            "surface_u": None if u is None else u.astype("<f4"), "info": info}


class LiveFit:
    def __init__(self, cloud, body_fit, max_model_age_s=1.5, refit_every=10 ** 9, process=None, body_args=None):
        self.cloud, self.fit = cloud, body_fit
        self.max_model_age_s, self.refit_every = max_model_age_s, refit_every
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.model = None             # newest model result (dict)
        self.result = None            # newest fitted result (dict)
        self.thread = None
        self.wanted_until = 0.0
        self.stats = {"fits": 0, "fit_ms": 0.0, "process": False}
        # body_args: (faces, joint_names, w_joint, w_vert, w), what a BodyFit is made from, for the fit process
        self.body_args = body_args
        self.use_process = (os.environ.get("FIT_PROCESS", "1") == "1") if process is None else process
        self.conn = self.child = None

    # ---- from the inference loop
    def set_model(self, owner, verts_cam_mm, jpos_cam_mm=None, jrot=None, mesh_id=0, extra=None):
        extra = extra or {}
        with self.lock:
            self.model = {"owner": owner, "verts": np.asarray(verts_cam_mm, np.float64), "jpos": jpos_cam_mm, "jrot": jrot,
                          "t": time.time(), "mesh_id": mesh_id, "extra": extra,
                          "cam_t_mm": np.asarray(extra.get("cam_t_mm", np.zeros(3)), np.float64)}
        self.wanted_until = time.time() + 2.0
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, name="live-fit", daemon=True)
            self.thread.start()

    def active(self, owner, within_s=0.6):
        r = self.result
        return r is not None and r["owner"] == owner and time.time() - r["t_done"] < within_s

    def wait(self, after_id, timeout=0.5):
        """Newest fitted result with id > after_id, or None after timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self.result is not None and self.result["id"] > after_id, timeout)
            r = self.result
        return r if (r is not None and r["id"] > after_id) else None

    # ---- the fit process
    def _start_process(self, cal):
        if not self.use_process or self.body_args is None:
            return False
        try:
            from multiprocessing.connection import Listener
            key = os.urandom(16)
            listener = Listener(("127.0.0.1", 0), authkey=key)
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Above-normal priority (Windows): the fit is the part that sets the latency of what is seen; the model loop, which
            # it competes with for the processor, only feeds it a prior a few times a second. One maths thread: its matrices
            # are small, and eight threads for each of them only get in the way of the model loop's.
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            flags = getattr(subprocess, "ABOVE_NORMAL_PRIORITY_CLASS", 0)
            self.child = subprocess.Popen([sys.executable, "-m", "depth.fitproc", str(listener.address[1]), key.hex()], cwd=repo,
                                          env=env, creationflags=flags)
            listener._listener._socket.settimeout(20.0)
            self.conn = listener.accept()
            listener.close()
            self.conn.send(("setup", self.body_args, self.cloud.rays, cal, self.refit_every))
            self.stats["process"] = True
            return True
        except Exception as exc:
            print(f"FIT_PROCESS not started ({type(exc).__name__}: {exc}); fitting in the server process", flush=True)
            self.conn = self.child = None
            self.use_process = False
            return False

    def _stop_process(self):
        try:
            if self.conn is not None:
                self.conn.send(("stop",)); self.conn.close()
        except Exception:
            pass
        self.conn = None
        if self.child is not None:
            try:
                self.child.wait(timeout=2)
            except Exception:
                self.child.kill()
        self.child = None

    # ---- the loop
    def _run(self):
        last_seq, fid, state, sent_mesh = -1, 0, FitState(), None
        try:
            while time.time() < self.wanted_until:
                item = self.cloud.get()
                cal = self.cloud.calibration
                if item is None or cal is None or item[0] == last_seq:
                    time.sleep(0.002)
                    continue
                with self.lock:
                    model = self.model
                if model is None or time.time() - model["t"] > self.max_model_age_s:
                    state = FitState(); sent_mesh = None
                    if self.conn is not None:
                        self.conn.send(("reset",))
                    time.sleep(0.005)
                    continue
                last_seq = item[0]
                t_depth = item[2] if len(item) > 2 else time.time()
                t0 = time.perf_counter()
                if self.conn is None and self.use_process:
                    self._start_process(cal)
                if self.conn is not None:
                    try:
                        key = (model["owner"], model["mesh_id"])
                        if key != sent_mesh:
                            self.conn.send(("model", {k: model[k] for k in ("owner", "mesh_id", "verts", "jpos", "jrot", "cam_t_mm")}))
                            sent_mesh = key
                        self.conn.send(("depth", item[1]))
                        out = self.conn.recv() if self.conn.poll(2.0) else None
                    except Exception as exc:                     # the process died: carry on in this one
                        print(f"FIT_PROCESS lost ({type(exc).__name__}: {exc}); fitting in the server process", flush=True)
                        self._stop_process(); self.use_process = False
                        out = None
                else:
                    depth = np.frombuffer(item[1], np.uint16).reshape(cal["h"], cal["w"])
                    frame = DepthFrame(depth, self.cloud.rays, cal["R"], cal["t"], cal["kd"], cal["dist"])
                    out = fit_one(state, self.fit, model, frame, self.refit_every)
                if out is None:
                    with self.cond:
                        self.result = None
                    continue
                fit_ms = (time.perf_counter() - t0) * 1000
                fid += 1
                out["info"]["fit_ms"] = round(fit_ms, 1)
                out.update(id=fid, owner=model["owner"], t_depth=t_depth, t_done=time.time(), depth_seq=last_seq,
                           mesh_id=model["mesh_id"], model_t=model["t"], extra=model["extra"])
                self.stats["fits"] += 1
                self.stats["fit_ms"] = 0.9 * self.stats["fit_ms"] + 0.1 * fit_ms if self.stats["fits"] > 1 else fit_ms
                with self.cond:
                    self.result = out
                    self.cond.notify_all()
        finally:
            self._stop_process()
            with self.cond:
                self.result = None
