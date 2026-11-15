"""Pose an atlas with a live MHR skin mesh (numpy reference implementation of anatomy.js).

Biomechanical model (see anatomy/rig.py):
  1. each rigid SEGMENT (pelvis, lumbar, lower/upper thorax, head, upper arms, thighs) gets a
     rest->live rigid transform from its supporting skin vertices (Kabsch);
  2. every vertebra is a rigid body whose rotation is a quaternion blend of the segments at its
     height; positions come from forward kinematics up the spine (spacing is preserved exactly),
     with a small end correction so the skull stays seated on C1;
  3. ribs / sternum / cartilages / clavicles / scapulae ride on the vertebrae they articulate
     with; pelvis, skull and limb bones follow their segment; all bones are rigid;
  4. SOFT tissue is skinned between the chain frames bracketing its height.
There is no organ physics and no claim of internal accuracy.
"""
import json
import numpy as np


class Atlas:
    def __init__(self, prefix):
        self.meta = json.load(open(prefix + ".json"))
        if self.meta.get("version") != 2:
            raise ValueError("atlas v2 required (rebuild with python -m anatomy.atlas_build ... --rebind)")
        blob = open(prefix + ".bin", "rb").read()
        self.segments = []
        for sg in self.meta["segments"]:
            n = sg["n"]
            self.segments.append({
                "name": sg["name"],
                "idx": np.frombuffer(blob, np.uint16, n, sg["idx"]).astype(np.int64),
                "rest": np.frombuffer(blob, np.float32, n * 3, sg["rest"]).reshape(n, 3).astype(np.float64)})
        self.structures = []
        for e in self.meta["structures"]:
            n = e["n_verts"]
            it = np.uint16 if e["idx_type"] == "u16" else np.uint32
            s = dict(e)
            s["pos"] = np.frombuffer(blob, np.float32, n * 3, e["pos"]).reshape(n, 3).astype(np.float64)
            s["faces"] = np.frombuffer(blob, it, e["n_tris"] * 3, e["idx"]).reshape(-1, 3).astype(np.int32)
            if e["bind"] == "soft":
                s["inf_idx"] = np.frombuffer(blob, np.uint8, n * 3, e["inf_idx"]).reshape(n, 3).astype(np.int64)
                s["inf_w"] = np.frombuffer(blob, np.float32, n * 3, e["inf_w"]).reshape(n, 3).astype(np.float64)
            self.structures.append(s)

    def names(self):
        return [s["name"] for s in self.structures]

    def segment_transforms(self, skin):
        """-> list of (q[w,x,y,z], R, c_rest, c_live), one per segment."""
        out = []
        for sg in self.segments:
            R, cs, cd = kabsch(sg["rest"], skin[sg["idx"]])
            out.append((rot_to_quat(R), R, cs, cd))
        return out

    def frames(self, skin):
        """-> (segment_frames, chain_frames, soft_frames). A frame is (q, R, c_rest, c_live)."""
        from . import rig
        T = self.segment_transforms(skin)
        chain = self.meta.get("chain", [])
        by = {s["name"]: s for s in self.structures}
        cf = []
        if chain:
            q, c = [], []
            for n in chain:
                ws = by[n]["bone_w"]
                ref = T[ws[0][0]][0]
                qq = sum(w * (T[sg][0] if T[sg][0] @ ref >= 0 else -T[sg][0]) for sg, w in ws)
                q.append(qq / np.linalg.norm(qq)); c.append(np.asarray(by[n]["centroid"]))
            # forward kinematics gives the local spacing/rotation; the skin-driven (segment-blended)
            # positions give the low-frequency placement. final = FK + smooth(skin - FK).
            sp = [sum(w * _apply(T[sg], c[i]) for sg, w in by[chain[i]]["bone_w"]) for i in range(len(chain))]
            cp = [sp[0]]
            for i in range(1, len(chain)):
                qm = q[i - 1] + (q[i] if q[i] @ q[i - 1] >= 0 else -q[i])
                cp.append(cp[-1] + quat_to_rot(qm / np.linalg.norm(qm)) @ (c[i] - c[i - 1]))
            d = np.array(sp) - np.array(cp)
            ker = np.exp(-np.arange(-6, 7) ** 2 / 18.0); ker /= ker.sum()
            dp = np.pad(d, ((6, 6), (0, 0)), mode="edge")
            ds = np.stack([np.convolve(dp[:, k], ker, mode="valid") for k in range(3)], 1)
            cp = np.array(cp) + ds
            cf = [(q[i], quat_to_rot(q[i]), c[i], cp[i]) for i in range(len(chain))]
        sf = [T[rig.SEG["pelvis"]]] + cf + [T[rig.SEG["head"]]] + [T[rig.SEG[n]] for n in ("uparm_l", "uparm_r", "thigh_l", "thigh_r")]
        return T, cf, sf

    def pose(self, skin):
        """skin: (V,3) live MHR vertices in the rest frame/units (metres). -> {name: (n,3)}."""
        T, cf, sf = self.frames(skin)
        out = {}
        for s in self.structures:
            b = s["bind"]
            if b == "bone":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(T[sg], w) for sg, w in s["bone_w"]])
            elif b == "vertebra":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(cf[s["chain_index"]], 1.0)])
            elif b == "attach":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(cf[i], w) for i, w in s["attach"]])
            else:
                out[s["name"]] = _pose_soft(s, sf)
        return out


def _apply(Tk, x):
    _, R, cs, cd = Tk
    return (x - cs) @ R.T + cd


def _rigid(pos, c, weighted_frames):
    """One rigid motion for a bone from one or more frames: quaternion-blended rotation, with the
    centroid carried by the weighted frame images (exact for a single frame)."""
    if len(weighted_frames) == 1:
        f = weighted_frames[0][0]
        return (pos - c) @ f[1].T + _apply(f, c)
    ref = weighted_frames[0][0][0]
    q = sum(w * (f[0] if f[0] @ ref >= 0 else -f[0]) for f, w in weighted_frames)
    R = quat_to_rot(q / np.linalg.norm(q))
    return (pos - c) @ R.T + sum(w * _apply(f, c) for f, w in weighted_frames)


def _pose_soft(s, frames):
    x = s["pos"]
    out = np.zeros_like(x)
    for k in range(3):
        wk = s["inf_w"][:, k]
        for fi in np.unique(s["inf_idx"][wk > 0, k]):
            m = (s["inf_idx"][:, k] == fi) & (wk > 0)
            out[m] += wk[m, None] * _apply(frames[fi], x[m])
    return out


def kabsch(src, dst, w=None):
    w = np.full(len(src), 1.0 / len(src)) if w is None else w / w.sum()
    cs, cd = w @ src, w @ dst
    H = ((src - cs) * w[:, None]).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cs, cd


# kept for the tests / parity check
weighted_kabsch = kabsch


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
