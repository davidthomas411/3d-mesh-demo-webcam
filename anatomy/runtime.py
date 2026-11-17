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


# ---------------------------------------------------------------------------------------------
# Physiologic distribution of trunk motion along the vertebral chain (spine="physio").
# AXIAL ROTATION per level, degrees to one side, relative to the inferior vertebra:
#   thoracic T1-T2 .. T12-L1: Fujimori et al. 2012, Spine, in vivo 3D CT of 13 volunteers in maximum trunk rotation (measured);
#   lumbar: Ochia 2006 (CT, 0.6-2.2 deg per level, largest L1-L2..L3-L4) and Pearcy 1984 (about 2 deg per level, L3-4/L4-5 slightly more):
#   the individual lumbar values below are chosen inside those ranges (ASSUMED), summing to ~8 deg (Narimani 2018: 7.5-9.2 deg).
# LATERAL BENDING / FLEXION shares are weaker: regional totals from Narimani 2018 (lumbar 17.4 / 55.4 deg, thoracic 25.5 / 19.9 deg),
#   lumbar lateral bending split 10:10:10:6:3 from Pearcy 1984, thoracic weighted toward the lower levels (Pan 2018 review: lower thoracic
#   contributes more in flexion and lateral bending), everything else uniform: ASSUMPTIONS, not measurements.
_AXIAL = {"L5": .8, "L4": 1.6, "L3": 2.0, "L2": 2.0, "L1": 1.8, "T12": .5, "T11": 1.3, "T10": 2.6, "T9": 2.7, "T8": 2.5,
          "T7": 2.3, "T6": 1.9, "T5": 1.8, "T4": 1.6, "T3": 1.4, "T2": 1.6, "T1": 1.2}
_TW = {f"T{k}": (0.8 if k <= 4 else 1.0 if k <= 8 else 1.3) for k in range(1, 13)}
_LAT = {**{n: 17.35 * w / 39.0 for n, w in zip(("L5", "L4", "L3", "L2", "L1"), (3, 6, 10, 10, 10))},
        **{n: 25.5 * w / sum(_TW.values()) for n, w in _TW.items()}}
_FLX = {**{n: 55.4 / 5 for n in ("L5", "L4", "L3", "L2", "L1")}, **{n: 19.9 * w / sum(_TW.values()) for n, w in _TW.items()}}


def physio_cumulative(chain):
    """name -> (Fx, Fy, Fz): cumulative fraction of the total pelvis-to-T1 rotation reached at each vertebra, per axis
    (x flexion/extension, y axial, z lateral bending in the pelvis frame), over the lumbar+thoracic vertebrae present."""
    names = [n for n in chain if n.replace("vertebrae_", "") in _AXIAL]
    tabs = [_FLX, _AXIAL, _LAT]
    tot = [sum(t[n.replace("vertebrae_", "")] for n in names) for t in tabs]
    run, out = [0.0, 0.0, 0.0], {}
    for n in names:
        k = n.replace("vertebrae_", "")
        for a, t in enumerate(tabs):
            run[a] += t[k]
        out[n] = tuple(run[a] / tot[a] for a in range(3))
    return out


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

    def segment_transforms(self, skin, joints=None, body_scale=True):
        """-> list of (q[w,x,y,z], R, c_rest, c_live, k), one per segment. k is the size of the person's segment relative to the
        neutral body (segment_scale; 1 when body_scale is off): everything a frame carries is scaled by k about c_rest, so the
        skeleton is sized to the body it sits in instead of staying atlas-sized.
        joints: optional (pos (J,3), rot (J,3,3)) MHR joint positions and global rotations in the atlas frame, as the estimator
        predicts them. A segment bound to a joint (atlas.json["mhr"], tools/bind_mhr_joints.py) then takes its ROTATION from that
        joint instead of a fit to its skin patch. Its position still comes from the patch (rest centroid -> live centroid): measured
        on real output, anchoring to the skin keeps the bones inside the body better than anchoring to the joint centres."""
        mhr = self.meta.get("mhr") if joints is not None else None
        out = []
        for sg in self.segments:
            j = mhr["segments"].get(sg["name"]) if mhr else None
            if j is not None:
                R = np.asarray(joints[1][j], np.float64) @ np.asarray(mhr["rest_rot"][str(j)]).reshape(3, 3).T
                q = rot_to_quat(R); R = quat_to_rot(q / np.linalg.norm(q))          # exactly orthonormal
                out.append((rot_to_quat(R), R, sg["rest"].mean(0), np.asarray(skin)[sg["idx"]].mean(0)))
                continue
            R, cs, cd = kabsch(sg["rest"], skin[sg["idx"]])
            out.append((rot_to_quat(R), R, cs, cd))
        return [f + (self.segment_scale(i, None, skin) if body_scale else 1.0,) for i, f in enumerate(out)]

    def frames(self, skin, breath=0.0, depth=1.0, joints=True, spine="skin", mhr_joints=None, body_scale=True):
        """-> (segment_frames, chain_frames, soft_frames). A frame is (q, R, c_rest, c_live, k): x -> k R (x - c_rest) + c_live.
        mhr_joints: optional (pos, rot) of the MHR skeleton; body_scale: size the skeleton to the person. See segment_transforms."""
        from . import rig
        T = self.segment_transforms(skin, mhr_joints, body_scale)
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
            if spine == "physio":                       # distribute the measured pelvis->thorax rotation by in-vivo per-level data
                from scipy.spatial.transform import Rotation as Rot
                Rp, Rt = T[rig.SEG["pelvis"]][1], T[rig.SEG["thorax_up"]][1]
                rv = Rot.from_matrix(Rp.T @ Rt).as_rotvec()
                for i, nm in enumerate(chain):
                    fr = physio_cumulative(chain).get(nm)
                    if fr is not None:
                        q[i] = rot_to_quat(Rp @ Rot.from_rotvec(np.array(fr) * rv).as_matrix())
            # forward kinematics gives the local spacing/rotation; the skin-driven (segment-blended)
            # positions give the low-frequency placement. final = FK + smooth(skin - FK).
            sp = [sum(w * _apply(T[sg], c[i]) for sg, w in by[chain[i]]["bone_w"]) for i in range(len(chain))]
            kc = [sum(w * _k(T[sg]) for sg, w in by[chain[i]]["bone_w"]) for i in range(len(chain))]   # size of each vertebra's level
            cp = [sp[0]]
            for i in range(1, len(chain)):
                qm = q[i - 1] + (q[i] if q[i] @ q[i - 1] >= 0 else -q[i])
                cp.append(cp[-1] + 0.5 * (kc[i - 1] + kc[i]) * (quat_to_rot(qm / np.linalg.norm(qm)) @ (c[i] - c[i - 1])))
            d = np.array(sp) - np.array(cp)
            ker = np.exp(-np.arange(-6, 7) ** 2 / 18.0); ker /= ker.sum()
            dp = np.pad(d, ((6, 6), (0, 0)), mode="edge")
            ds = np.stack([np.convolve(dp[:, k], ker, mode="valid") for k in range(3)], 1)
            cp = np.array(cp) + ds
            cf = [(q[i], quat_to_rot(q[i]), c[i], cp[i], kc[i]) for i in range(len(chain))]
        sf = [T[rig.SEG["pelvis"]]] + cf + [T[rig.SEG["head"]]] + [T[rig.SEG[n]] for n in ("uparm_l", "uparm_r", "thigh_l", "thigh_r")]
        F = self._joint_frames(T, cf, chain, by, breath, depth, mhr_joints=mhr_joints) if joints else self._legacy_frames(T, cf, chain)
        if joints and mhr_joints is not None and self.meta.get("mhr"):   # limb bones sized to this person's bone lengths
            for bone, lb in self.meta["mhr"].get("limbs", {}).items():
                if bone in F:
                    n = np.linalg.norm(np.asarray(mhr_joints[0][lb["child"]], np.float64) - np.asarray(mhr_joints[0][lb["joint"]], np.float64))
                    # the scale is uniform, so a bone lengthened for a long thin arm would also get thicker and leave the skin:
                    # take the smaller of the length ratio and the limb's girth ratio (the frame's own scale)
                    F[bone] = F[bone][:4] + (min(float(np.clip(n / lb["rest_len"], 0.8, 1.25)), _k(F[bone])),)
        return T, cf, sf, F

    def segment_scale(self, seg, T, skin, lo=0.65, hi=1.35):
        """Size of a segment's live skin patch relative to its rest patch (ratio of RMS radii about the centroids), clamped. The
        neutral MHR body is one size and the estimator predicts each person's own shape, so a bone would otherwise stay
        atlas-sized in a body that is bigger or smaller. (T is unused, kept for callers.)"""
        sg = self.segments[seg]; live = np.asarray(skin)[sg["idx"]]
        a = ((sg["rest"] - sg["rest"].mean(0)) ** 2).sum(); b = ((live - live.mean(0)) ** 2).sum()
        return float(np.clip(np.sqrt(b / max(a, 1e-12)), lo, hi))

    def _legacy_frames(self, T, cf, chain):
        """The model stage BEFORE joint rules: ribs rigid on their vertebra, sternum/clavicle/scapula attached to the
        vertebrae they sit near, humerus/femur/skull following their segment. No pins, no limits, no breathing."""
        import re
        from . import rig
        idx = {n: i for i, n in enumerate(chain)}
        F = {}
        for s_ in self.structures:
            n = s_["name"]
            if s_["bind"] != "joint":
                continue
            m = re.match(r"rib_(?:left|right)_(\d+)$", n)
            if m and f"vertebrae_T{m.group(1)}" in idx:
                F[n] = cf[idx[f"vertebrae_T{m.group(1)}"]]
                continue
            att = rig.attach_for(n, chain)
            if att:
                F[n] = _blend_frame(np.asarray(s_["centroid"]), [(cf[i], w) for i, w in att])
                continue
            side = "l" if n.endswith("_left") else "r"
            if n.startswith("humerus_"):
                F[n] = T[rig.SEG[f"uparm_{side}"]]
            elif n.startswith("femur_"):
                F[n] = T[rig.SEG[f"thigh_{side}"]]
            elif n == "skull":
                F[n] = T[rig.SEG["head"]]
        return F

    def _joint_frames(self, T, cf, chain, by, breath, depth, glide=True, mhr_joints=None):
        """Body frames (q, R, J, pJ, k): x' = k R (x - J) + pJ, for every joint-driven bone.
        With mhr_joints and an atlas whose arm bones are seated on the MHR skeleton (tools/seat_limbs_on_mhr.py), the humerus is
        defined on the model's own skeleton: its head rides the live MHR shoulder joint. The scapula keeps the ball in its socket
        (it is pinned at that joint) and the clavicle reaches from the sternum to the scapula, so the chain is solved from the arm
        back to the trunk instead of from the trunk out to the arm."""
        from . import rig
        from scipy.spatial.transform import Rotation as Rot
        SEG = rig.SEG
        J = {j["name"]: j for j in self.meta.get("joints", [])}
        F = {}
        mk = lambda R, Jc, pJ, k=1.0: (rot_to_quat(R), R, np.asarray(Jc), pJ, k)
        for j in J.values():                                   # ribs: costovertebral joint + respiratory rotation
            if j["type"] != "costovertebral" or j["parent"] not in chain:
                continue
            fv = cf[chain.index(j["parent"])]
            Jc = np.asarray(j["center"])
            ang = np.radians(j["resp_deg"] * depth * breath) * j["resp_sign"]
            Rr = Rot.from_rotvec(np.asarray(j["axis"]) * ang).as_matrix()
            F[j["child"]] = mk(fv[1] @ Rr, Jc, _apply(fv, Jc), _k(fv))
        if "sternum" in by and by["sternum"].get("attach_ribs"):
            wf = [(F[r], w) for r, w in by["sternum"]["attach_ribs"] if r in F]
            F["sternum"] = _blend_frame(np.asarray(by["sternum"]["centroid"]), wf)
        S = F.get("sternum")
        Rthup = T[SEG["thorax_up"]][1]
        mhr = self.meta.get("mhr") if mhr_joints is not None else None
        on_skeleton = bool(mhr and mhr.get("limb_seated"))
        for k, side in (("l", "left"), ("r", "right")):
            Rarm, Rthigh = T[SEG[f"uparm_{k}"]][1], T[SEG[f"thigh_{k}"]][1]
            lb = mhr["limbs"].get(f"humerus_{side}") if on_skeleton else None
            if S is not None and lb and all(f"{n}_{k}" in J for n in ("sc", "ac", "gh")):
                sc, ac, gh = J[f"sc_{k}"], J[f"ac_{k}"], J[f"gh_{k}"]
                Jsc, Jac, Jgh = (np.asarray(j["center"]) for j in (sc, ac, gh))
                head = np.asarray(mhr_joints[0][lb["joint"]], np.float64)      # live MHR shoulder joint = centre of the humeral head
                ks = _k(S)
                # scapula: glenoid on the humeral head; a third of the arm's elevation, then glide on the rib cage about that pin
                rv = Rot.from_matrix(Rthup.T @ Rarm).as_rotvec()
                Rsc = S[1] @ rig.limit_rotation(Rot.from_rotvec(rv / 3.0).as_matrix(), ac["limit_deg"])
                Rsc = _glide(ac, Rsc, Jgh, head, S, by[ac["child"]]["pos"]) if glide else Rsc
                F[ac["child"]] = mk(Rsc, Jgh, head, ks)
                # clavicle: from the sternoclavicular joint to where the scapula put the acromioclavicular joint
                pJ = _apply(S, Jsc); tgt = _apply(F[ac["child"]], Jac) - pJ
                Rcl = _aim(S[1] @ (Jac - Jsc), tgt, sc["limit_deg"]) @ S[1]
                kcl = float(np.clip(np.linalg.norm(tgt) / np.linalg.norm(Jac - Jsc), 0.75 * ks, 1.35 * ks))
                F[sc["child"]] = mk(Rcl, Jsc, pJ, kcl)
                # humerus: on the model's skeleton
                Rh = Rsc @ rig.limit_rotation(Rsc.T @ Rarm, gh["limit_deg"])
                F[gh["child"]] = mk(Rh, Jgh, head, _k(T[SEG[f"uparm_{k}"]]))
            elif S is not None and f"sc_{k}" in J:
                sc, ac = J[f"sc_{k}"], J.get(f"ac_{k}")
                Jsc = np.asarray(sc["center"]); pJ = _apply(S, Jsc)
                # clavicle: aim its lateral end (AC joint) at where the shoulder cap went, about the SC joint
                tgt = _apply(T[SEG[f"shoulder_{k}"]], np.asarray(ac["center"])) - pJ
                a = S[1] @ (np.asarray(ac["center"]) - Jsc)
                Rcl = _aim(a, tgt, sc["limit_deg"]) @ S[1]
                F[sc["child"]] = mk(Rcl, Jsc, pJ, _k(S))
                # scapula: pinned at the AC joint; rotates with a third of the arm's elevation (scapulohumeral rhythm)
                Jac = np.asarray(ac["center"]); pJac = _apply(F[sc["child"]], Jac)
                rv = Rot.from_matrix(Rthup.T @ Rarm).as_rotvec()
                Rrel = rig.limit_rotation(Rot.from_rotvec(rv / 3.0).as_matrix(), ac["limit_deg"])
                Rsc = S[1] @ Rrel
                Rsc = _glide(ac, Rsc, Jac, pJac, S, by[ac["child"]]["pos"]) if glide else Rsc
                F[ac["child"]] = mk(Rsc, Jac, pJac, _k(S))
                gh = J.get(f"gh_{k}")
                if gh:                                          # humerus: ball pinned in the glenoid
                    Jgh = np.asarray(gh["center"])
                    Rh = Rsc @ rig.limit_rotation(Rsc.T @ Rarm, gh["limit_deg"])
                    F[gh["child"]] = mk(Rh, Jgh, _apply(F[ac["child"]], Jgh), _k(T[SEG[f"uparm_{k}"]]))
            hip = J.get(f"hip_{k}")
            if hip:                                             # femur: ball pinned in the acetabulum
                Tp = T[SEG["pelvis"]]; Jh = np.asarray(hip["center"])
                Rf = Tp[1] @ rig.limit_rotation(Tp[1].T @ Rthigh, hip["limit_deg"])
                F[hip["child"]] = mk(Rf, Jh, _apply(Tp, Jh), _k(T[SEG[f"thigh_{k}"]]))
        if "ao" in J and J["ao"]["parent"] in chain:           # skull on C1
            c1 = cf[chain.index(J["ao"]["parent"])]; Jao = np.asarray(J["ao"]["center"])
            Rh = T[SEG["head"]][1]
            # The skull rides the head: it turns with it (within the joint's range on C1) and is carried by the head frame itself,
            # the same rule as the humerus on its joint. Pinning it to C1 instead let the spine's smoothing drag it off the head.
            Th = T[SEG["head"]]
            F["skull"] = mk(c1[1] @ rig.limit_rotation(c1[1].T @ Rh, J["ao"]["limit_deg"]), Jao, _apply(Th, Jao), _k(Th))
        return F

    def _cartilage(self, s, F):
        """Costal cartilage is flexible, not a rigid bone: each vertex is carried by the rib it is nearest to at its rib end and by
        the sternum at its sternal end, blended by rest-pose distance. Both ends therefore stay joined to the bones they connect
        (no gap at the costochondral or sternocostal junction) however the ribs and sternum move relative to each other."""
        if getattr(self, "_cart", None) is None:
            from scipy.spatial import cKDTree
            ribs = [t for t in self.structures if t["name"].startswith("rib_") and t["name"] in F]
            rp = np.vstack([t["pos"] for t in ribs]); rown = np.concatenate([[t["name"]] * len(t["pos"]) for t in ribs])
            sp = next(t["pos"] for t in self.structures if t["name"] == "sternum")
            dr, ir = cKDTree(rp).query(s["pos"]); ds, _ = cKDTree(sp).query(s["pos"])
            self._cart = (rown[ir], dr / np.maximum(dr + ds, 1e-9))
        own, w = self._cart
        out = np.empty_like(s["pos"]); xs = _apply(F["sternum"], s["pos"])
        for r in np.unique(own):
            m = own == r
            out[m] = (1 - w[m, None]) * _apply(F[r], s["pos"][m]) + w[m, None] * xs[m]
        return out

    def pose(self, skin, breath=0.0, depth=1.0, joints=True, spine="skin", mhr_joints=None, body_scale=True):
        """skin: (V,3) live MHR vertices in the rest frame/units (metres). breath: 0 (end-expiration) .. 1
        (end-inspiration), depth: 1 = quiet breathing. joints=False reproduces the model stage before joint rules;
        spine="physio" distributes trunk rotation by in-vivo per-level data (numpy only; the browser port is pending).
        -> {name: (n,3)}."""
        T, cf, sf, F = self.frames(skin, breath, depth, joints, spine, mhr_joints, body_scale)
        out = {}
        for s in self.structures:
            b = s["bind"]
            if b == "joint":
                out[s["name"]] = _apply(F[s["name"]], s["pos"])
            elif b == "frame":                               # rigid with another body's frame (brain with the skull)
                out[s["name"]] = _apply(F[s["frame"]], s["pos"])
            elif s["name"] == "costal_cartilages" and joints and "sternum" in F:
                out[s["name"]] = self._cartilage(s, F)      # flexible link between the ribs' ends and the sternum
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


def ellipsoid_gap(x, c, r):
    """Approximate signed distance (m) of points x to an axis-aligned ellipsoid (positive outside) and its outward unit normal."""
    q = (x - c) / r
    f = np.sqrt((q ** 2).sum(1))
    d = np.linalg.norm(x - c, axis=1) * (1 - 1 / np.maximum(f, 1e-9))
    n = (x - c) / r ** 2
    return d, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _glide(ac, Rsc, Jac, pJac, S, scap_pos, iters=6):
    """Scapulothoracic contact: the scapula glides on the rib cage. Swing it about its acromioclavicular pin until the mean gap of its
    contact patch to the thorax ellipsoid is back to the rest value (fixed iteration count, mirrored in anatomy.js)."""
    from scipy.spatial.transform import Rotation as Rot
    if "patch" not in ac:
        return Rsc
    P = scap_pos[ac["patch"]]; c = np.asarray(ac["ellipsoid"]["center"]); r = np.asarray(ac["ellipsoid"]["radii"])
    ks = _k(S)
    for _ in range(iters):
        y = ks * (P - Jac) @ Rsc.T + pJac                             # posed contact patch (scaled with the thorax)
        x = S[2] + (y - S[3]) @ S[1] / ks                             # into the sternum's rest frame: c + R^T (y - cp) / k
        d, n = ellipsoid_gap(x, c, r)
        nb = (n @ S[1].T).mean(0); nb = nb / max(np.linalg.norm(nb), 1e-12)
        lever = y.mean(0) - pJac
        # mean gap back to rest, and no deeper into the thorax than the rest-pose closest point (penetration term)
        shift = (d.mean() - ac["gap0"]) - max(ac["gapmin0"] - d.min(), 0.0)
        w = np.cross(lever, -shift * ks * nb) / max(lever @ lever, 1e-9)
        a = np.linalg.norm(w)
        if a > 0.15:
            w = w * (0.15 / a)
        Rsc = Rot.from_rotvec(0.8 * w).as_matrix() @ Rsc
    return Rsc


def _blend_frame(centroid, wf):
    """Frame (q, R, c, cp) of a bone carried by several frames: blended rotation, averaged centroid image."""
    ref = wf[0][0][0]
    q = sum(w * (f[0] if f[0] @ ref >= 0 else -f[0]) for f, w in wf)
    q = q / np.linalg.norm(q)
    return (q, quat_to_rot(q), centroid, sum(w * _apply(f, centroid) for f, w in wf), sum(w * _k(f) for f, w in wf) / sum(w for _, w in wf))


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


def _k(f):
    """Uniform scale of a frame (1 if it carries none)."""
    return f[4] if len(f) > 4 else 1.0


def _apply(Tk, x):
    """Rigid frame (q, R, c_rest, c_live); an optional fifth entry is a uniform scale about c_rest (the skull)."""
    _, R, cs, cd = Tk[:4]
    k = Tk[4] if len(Tk) > 4 else 1.0
    return k * (x - cs) @ R.T + cd


def _rigid(pos, c, weighted_frames):
    """One rigid motion for a bone from one or more frames: quaternion-blended rotation, with the
    centroid carried by the weighted frame images (exact for a single frame)."""
    if len(weighted_frames) == 1:
        f = weighted_frames[0][0]
        return _k(f) * (pos - c) @ f[1].T + _apply(f, c)
    ref = weighted_frames[0][0][0]
    q = sum(w * (f[0] if f[0] @ ref >= 0 else -f[0]) for f, w in weighted_frames)
    R = quat_to_rot(q / np.linalg.norm(q))
    k = sum(w * _k(f) for f, w in weighted_frames) / sum(w for _, w in weighted_frames)
    return k * (pos - c) @ R.T + sum(w * _apply(f, c) for f, w in weighted_frames)


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
