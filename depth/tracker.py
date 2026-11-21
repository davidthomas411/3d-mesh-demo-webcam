"""The low-latency path: every depth frame, fit the newest body mesh to it and publish the result.

Two clocks run in this demo. The RGB model produces a posed mesh from a picture that is, by the time it is shown, 150-350 ms
old, five to seven times a second. The depth camera delivers the true surface thirty times a second, a few tens of
milliseconds old. Showing the model's mesh next to the live depth makes the mesh trail behind the person.

Here the model's mesh is no longer what is shown. It is the SHAPE AND POSE PRIOR: each new depth frame is fitted with the
newest model mesh (depth/fit.py), starting from the previous frame's fit, and the fitted mesh is what goes to the page. What the
viewer sees is therefore as old as the depth frame plus one fit (about 35 ms), and it sits on the measured surface. When the
model delivers a new mesh, the next fit simply starts from it with the previous part transforms: the mesh carries what the
picture knows (pose, limbs the depth camera cannot see), the transforms carry where the body is now.

A fit that fails (nobody in the depth image) or drifts (its distance stage disagrees with the tracked one) starts again from
scratch on the next frame.
"""
import threading
import time

import numpy as np

from .fit import DepthFrame


class LiveFit:
    def __init__(self, cloud, body_fit, max_model_age_s=1.5, refit_every=45):
        self.cloud, self.fit = cloud, body_fit
        self.max_model_age_s, self.refit_every = max_model_age_s, refit_every
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.model = None             # newest model result: dict(owner, verts_mm, jpos_mm, jrot, t, mesh_id, extra)
        self.result = None            # newest fitted result (dict)
        self.thread = None
        self.wanted_until = 0.0
        self.stats = {"fits": 0, "fit_ms": 0.0}

    # ---- from the inference loop
    def set_model(self, owner, verts_cam_mm, jpos_cam_mm=None, jrot=None, mesh_id=0, extra=None):
        with self.lock:
            self.model = {"owner": owner, "verts": np.asarray(verts_cam_mm, np.float64), "jpos": jpos_cam_mm, "jrot": jrot,
                          "t": time.time(), "mesh_id": mesh_id, "extra": extra or {}}
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

    # ---- the fit loop
    def _run(self):
        last_seq, M, owner, n_since_scratch, fid, track = -1, None, None, 0, 0, {}
        while time.time() < self.wanted_until:
            item = self.cloud.get()
            cal = self.cloud.calibration
            if item is None or cal is None or item[0] == last_seq:
                time.sleep(0.002)
                continue
            with self.lock:
                model = self.model
            if model is None or time.time() - model["t"] > self.max_model_age_s:
                M = None
                time.sleep(0.005)
                continue
            last_seq = item[0]
            t_depth = item[2] if len(item) > 2 else time.time()
            t0 = time.perf_counter()
            depth = np.frombuffer(item[1], np.uint16).reshape(cal["h"], cal["w"])
            frame = DepthFrame(depth, self.cloud.rays, cal["R"], cal["t"], cal["kd"], cal["dist"])
            if model["owner"] != owner or n_since_scratch >= self.refit_every:
                M, owner, n_since_scratch, track = None, model["owner"], 0, {}
            V = model["verts"]
            if model["mesh_id"] != track.get("mesh_id"):
                # A new model mesh: its shape differs a little from the last one (the model's own frame-to-frame noise), so
                # the surface offsets worked out for the last one do not belong to it. They start afresh on this frame and
                # are averaged over the frames that use this mesh.
                track["u"] = None; track["mesh_id"] = model["mesh_id"]
            X, Mn, info = self.fit.fit(V, frame, init=M, pose_iters=1 if M is not None else 2, track=track)
            if Mn is None and M is not None:                     # tracking lost: once more from scratch on this frame
                track = {}
                X, Mn, info = self.fit.fit(V, frame, init=None, pose_iters=2, track=track)
            M = Mn
            n_since_scratch += 1
            if Mn is None:
                info["fit_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                with self.cond:
                    self.result = None
                continue
            # the body origin the page places the mesh at: where the pelvis part carries the model's own origin
            t_model = model["extra"].get("cam_t_mm", np.zeros(3))
            t_new = Mn[0, :3, :3] @ t_model + Mn[0, :3, 3]
            joints = None
            if model["jpos"] is not None:
                P, Rj = self.fit.move_joints(model["jpos"], model["jrot"], Mn)
                joints = (P - t_new, Rj)
            fit_ms = (time.perf_counter() - t0) * 1000
            fid += 1
            info.update(fit=True, fit_ms=round(fit_ms, 1), samples=info.get("surface_vertices", 0),
                        scale=info.get("part_scale", [1.0])[0], tracked=n_since_scratch > 1)
            out = {"id": fid, "owner": owner, "verts_m": ((X - t_new) / 1000.0).astype("<f4"), "cam_t_m": (t_new / 1000.0).astype(np.float32),
                   "joints": joints, "info": info, "t_depth": t_depth, "t_done": time.time(), "depth_seq": last_seq,
                   "mesh_id": model["mesh_id"], "model_t": model["t"], "extra": model["extra"]}
            self.stats["fits"] += 1; self.stats["fit_ms"] = 0.9 * self.stats["fit_ms"] + 0.1 * fit_ms if self.stats["fits"] > 1 else fit_ms
            with self.cond:
                self.result = out
                self.cond.notify_all()
        with self.cond:
            self.result = None
