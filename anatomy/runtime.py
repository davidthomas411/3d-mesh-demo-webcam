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
  5. JOINTS (meta["joints"]): a child bone rotates about its joint centre and that centre is pinned to the
     parent's image of it, so nothing can pull a ball out of its socket. Rotations are clamped to a range of motion.
     ribs   : costovertebral joint on their vertebra, plus the respiratory rotation about the neck axis
     sternum: carried by the ribs it is joined to (so it rises with breathing)
     clavicle: pivots about the sternoclavicular joint so its lateral end follows the shoulder cap (shrug, protraction)
     scapula : pinned at the acromioclavicular joint; follows about a third of the arm's elevation (scapulohumeral rhythm)
     humerus : ball joint pinned in the glenoid, rotation from the upper-arm skin
     femur   : ball joint pinned in the acetabulum, rotation from the thigh skin
     skull   : pinned on C1 (atlanto-occipital joint), rotation from the head skin
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

    def frames(self, skin, breath=0.0, depth=1.0):
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
        F = self._joint_frames(T, cf, chain, by, breath, depth)
        return T, cf, sf, F

    def _joint_frames(self, T, cf, chain, by, breath, depth):
        """Body frames (q, R, J, pJ): x' = R (x - J) + pJ, for every joint-driven bone."""
        from . import rig
        from scipy.spatial.transform import Rotation as Rot
        SEG = rig.SEG
        J = {j["name"]: j for j in self.meta.get("joints", [])}
        F = {}
        mk = lambda R, Jc, pJ: (rot_to_quat(R), R, np.asarray(Jc), pJ)
        for j in J.values():                                   # ribs: costovertebral joint + respiratory rotation
            if j["type"] != "costovertebral" or j["parent"] not in chain:
                continue
            fv = cf[chain.index(j["parent"])]
            Jc = np.asarray(j["center"])
            ang = np.radians(j["resp_deg"] * depth * breath) * j["resp_sign"]
            Rr = Rot.from_rotvec(np.asarray(j["axis"]) * ang).as_matrix()
            F[j["child"]] = mk(fv[1] @ Rr, Jc, _apply(fv, Jc))
        if "sternum" in by and by["sternum"].get("attach_ribs"):
            wf = [(F[r], w) for r, w in by["sternum"]["attach_ribs"] if r in F]
            F["sternum"] = _blend_frame(np.asarray(by["sternum"]["centroid"]), wf)
        S = F.get("sternum")
        Rthup = T[SEG["thorax_up"]][1]
        for k, side in (("l", "left"), ("r", "right")):
            Rarm, Rthigh = T[SEG[f"uparm_{k}"]][1], T[SEG[f"thigh_{k}"]][1]
            if S is not None and f"sc_{k}" in J:
                sc, ac = J[f"sc_{k}"], J.get(f"ac_{k}")
                Jsc = np.asarray(sc["center"]); pJ = _apply(S, Jsc)
                # clavicle: aim its lateral end (AC joint) at where the shoulder cap went, about the SC joint
                tgt = _apply(T[SEG[f"shoulder_{k}"]], np.asarray(ac["center"])) - pJ
                a = S[1] @ (np.asarray(ac["center"]) - Jsc)
                Rcl = _aim(a, tgt, sc["limit_deg"]) @ S[1]
                F[sc["child"]] = mk(Rcl, Jsc, pJ)
                # scapula: pinned at the AC joint; rotates with a third of the arm's elevation (scapulohumeral rhythm)
                Jac = np.asarray(ac["center"]); pJac = _apply(F[sc["child"]], Jac)
                rv = Rot.from_matrix(Rthup.T @ Rarm).as_rotvec()
                Rrel = rig.limit_rotation(Rot.from_rotvec(rv / 3.0).as_matrix(), ac["limit_deg"])
                Rsc = S[1] @ Rrel
                F[ac["child"]] = mk(Rsc, Jac, pJac)
                gh = J.get(f"gh_{k}")
                if gh:                                          # humerus: ball pinned in the glenoid
                    Jgh = np.asarray(gh["center"])
                    Rh = Rsc @ rig.limit_rotation(Rsc.T @ Rarm, gh["limit_deg"])
                    F[gh["child"]] = mk(Rh, Jgh, _apply(F[ac["child"]], Jgh))
            hip = J.get(f"hip_{k}")
            if hip:                                             # femur: ball pinned in the acetabulum
                Tp = T[SEG["pelvis"]]; Jh = np.asarray(hip["center"])
                Rf = Tp[1] @ rig.limit_rotation(Tp[1].T @ Rthigh, hip["limit_deg"])
                F[hip["child"]] = mk(Rf, Jh, _apply(Tp, Jh))
        if "ao" in J and J["ao"]["parent"] in chain:           # skull on C1
            c1 = cf[chain.index(J["ao"]["parent"])]; Jao = np.asarray(J["ao"]["center"])
            Rh = T[SEG["head"]][1]
            F["skull"] = mk(c1[1] @ rig.limit_rotation(c1[1].T @ Rh, J["ao"]["limit_deg"]), Jao, _apply(c1, Jao))
        return F

    def pose(self, skin, breath=0.0, depth=1.0):
        """skin: (V,3) live MHR vertices in the rest frame/units (metres). breath: 0 (end-expiration) .. 1
        (end-inspiration), depth: 1 = quiet breathing. -> {name: (n,3)}."""
        T, cf, sf, F = self.frames(skin, breath, depth)
        out = {}
        for s in self.structures:
            b = s["bind"]
            if b == "joint":
                out[s["name"]] = _apply(F[s["name"]], s["pos"])
            elif b == "attach_ribs":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(F[r], w) for r, w in s["attach_ribs"] if r in F])
            elif b == "bone":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(T[sg], w) for sg, w in s["bone_w"]])
            elif b == "vertebra":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(cf[s["chain_index"]], 1.0)])
            elif b == "attach":
                out[s["name"]] = _rigid(s["pos"], np.asarray(s["centroid"]), [(cf[i], w) for i, w in s["attach"]])
            else:
                out[s["name"]] = _pose_soft(s, sf)
        return out


def _blend_frame(centroid, wf):
    """Frame (q, R, c, cp) of a bone carried by several frames: blended rotation, averaged centroid image."""
    ref = wf[0][0][0]
    q = sum(w * (f[0] if f[0] @ ref >= 0 else -f[0]) for f, w in wf)
    q = q / np.linalg.norm(q)
    return (q, quat_to_rot(q), centroid, sum(w * _apply(f, centroid) for f, w in wf))


def _aim(a, b, max_deg):
    """Shortest-arc rotation taking vector a onto the direction of b, clamped to max_deg."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb
    v = np.cross(a, b); c = float(np.clip(a @ b, -1, 1)); sn = np.linalg.norm(v)
    if sn < 1e-9:
        return np.eye(3)
    ang = min(np.arctan2(sn, c), np.radians(max_deg))
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(v / sn * ang).as_matrix()


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
