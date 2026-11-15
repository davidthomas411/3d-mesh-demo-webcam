"""Biomechanical rig for the anatomy atlas (MHR rest frame: metres, y down, +x person-left).

Model
-----
* The body is a chain of RIGID SEGMENTS: pelvis, lumbar, lower thorax, upper thorax, head, upper arms, thighs
  (MHR distributes spine rotation over several joints, so a single rigid thorax is not enough). Each
  segment's rest->live rigid transform is estimated at runtime from a fixed set of MHR skin
  vertices that move with it (weighted Kabsch).
* BONES are rigid. A bone follows one segment (ribs, sternum, scapulae, clavicles -> thorax;
  hip bones, sacrum -> pelvis; skull -> head; humerus -> upper arm; femur -> thigh). Spinal
  vertebrae follow a blend of neighbouring segments chosen by their height on the spine chain
  (lumbar: pelvis<->thorax, cervical: thorax<->head), so bending and axial twist are
  distributed along the spine while each vertebra stays a rigid body.
* SOFT TISSUE (organs, vessels, muscles, airway, brain) uses linear blend skinning with
  per-vertex weights from the same chain, so it deforms continuously between segments.
* Limb bones from the source CT are pose-normalised: swung about their joint centre onto the
  MHR limb axis, because the CT has the arms down and the MHR rest pose is a T-pose.

Everything here is data prep; runtime (anatomy.js / runtime.py) only consumes the results.
"""
import numpy as np

CHAIN = ["pelvis", "lumbar", "thorax_low", "thorax_up", "head"]        # spine chain, bottom to top
SEGMENTS = CHAIN + ["uparm_l", "uparm_r", "thigh_l", "thigh_r", "shoulder_l", "shoulder_r"]
SEG = {n: i for i, n in enumerate(SEGMENTS)}
KP = {"lSh": 5, "rSh": 6, "lEl": 7, "rEl": 8, "lHip": 9, "rHip": 10, "lKn": 11, "rKn": 12, "neck": 69}


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _near_segment(v, a, b, radius, u_lo, u_hi):
    ab = b - a
    u = ((v - a) @ ab) / (ab @ ab)
    d = np.linalg.norm(v - (a + np.clip(u, 0, 1)[:, None] * ab), axis=1)
    return (d < radius) & (u > u_lo) & (u < u_hi)


def spine_levels(centroids, joints):
    """y (down-positive) of each chain segment's centre, taken from the atlas vertebrae."""
    hip_y = 0.5 * (joints[KP["lHip"]][1] + joints[KP["rHip"]][1])
    neck_y = joints[KP["neck"]][1]
    g = lambda n, default: float(centroids[n][1]) if n in centroids else default
    return {"pelvis": hip_y - 0.05, "lumbar": g("vertebrae_L3", hip_y - 0.19), "thorax_low": g("vertebrae_T9", hip_y - 0.34),
            "thorax_up": g("vertebrae_T3", neck_y + 0.09), "C7": g("vertebrae_C7", neck_y - 0.01),
            "C2": g("vertebrae_C2", neck_y - 0.12), "hip_y": float(hip_y)}


def build_segments(mhr_v, joints, lv):
    """Skin vertex ids that move with each rigid segment, from rest-pose MHR geometry."""
    j = {k: joints[i] for k, i in KP.items()}
    hip_y = lv["hip_y"]
    x, y = mhr_v[:, 0], mhr_v[:, 1]
    band = lambda yc, hh, xl: np.flatnonzero((np.abs(y - yc) < hh) & (np.abs(x) < xl))
    out = {
        "head": np.flatnonzero(y < j["neck"][1] - 0.03),
        "pelvis": np.flatnonzero((y > hip_y - 0.12) & (y < hip_y + 0.03) & (np.abs(x) < 0.20)),
        "lumbar": band(lv["lumbar"], 0.05, 0.17),
        "thorax_low": band(lv["thorax_low"], 0.05, 0.15),
        "thorax_up": band(lv["thorax_up"], 0.05, 0.12),     # narrow: keep the armpits out so arm motion cannot leak in
    }
    for s, side in (("l", +1), ("r", -1)):
        # shoulder girdle (clavicle/scapula motion: shrug, protraction): the upper cap of the shoulder, which
        # follows the girdle while the deltoid below it follows the arm
        cap = j[f"{s}Sh"] + np.array([0.0, -0.02, 0.0])
        out[f"shoulder_{s}"] = np.flatnonzero((np.linalg.norm(mhr_v - cap, axis=1) < 0.065) & (y < j[f"{s}Sh"][1] + 0.01))
        out[f"uparm_{s}"] = np.flatnonzero(
            _near_segment(mhr_v, j[f"{s}Sh"], j[f"{s}El"], 0.06, 0.25, 0.9) & (x * side > 0.19))
        out[f"thigh_{s}"] = np.flatnonzero(
            _near_segment(mhr_v, j[f"{s}Hip"], j[f"{s}Kn"], 0.09, 0.25, 0.9) & (y > hip_y + 0.07) & (x * side > 0))
    return out


def chain_weights_y(y, lv):
    """Weights (n, len(CHAIN)) from height alone along the spine chain. y down: smaller = higher.
    Piecewise smoothstep between consecutive segment centres; head ramps in over C7->C2."""
    y = np.atleast_1d(np.asarray(y, float))
    W = np.zeros((len(y), len(CHAIN)))
    centres = [lv["pelvis"], lv["lumbar"], lv["thorax_low"], lv["thorax_up"]]
    for i, yc in enumerate(centres):                       # tent functions between neighbours
        up = centres[i + 1] if i + 1 < len(centres) else None
        dn = centres[i - 1] if i > 0 else None
        w = np.zeros(len(y))
        if up is not None:
            m = (y <= yc) & (y > up); w[m] = 1 - _smooth((yc - y[m]) / (yc - up))
        if dn is not None:
            m = (y > yc) & (y < dn); w[m] = 1 - _smooth((y[m] - yc) / (dn - yc))
        w[(y >= yc) if dn is None else (y == yc)] = 1.0 if dn is None else 1.0
        if i == 0: w[y >= yc] = 1.0
        if i == len(centres) - 1: w[y <= yc] = 1.0
        W[:, i] = w
    head = _smooth((lv["C7"] - y) / max(lv["C7"] - lv["C2"], 1e-3))
    W[:, :4] *= (1 - head)[:, None]
    W[:, 4] = head
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    return W


_BONE_SEG = [(r"^skull$", "head"), (r"^(sacrum|hip_)", "pelvis")]
_CHAIN_BONE = r"^(rib_|sternum|costal_cartilages|clavicula_|scapula_)"


def side_of(name):
    return "l" if name.endswith("_left") else "r" if name.endswith("_right") else None


def classify(name):
    """-> ('bone', fixed_segment|None) | ('bone_spine', None) | ('soft', None)."""
    import re
    if name == "vertebrae_S1":                      # sacral: part of the pelvis, not the mobile spine
        return "bone", "pelvis"
    if name.startswith("vertebrae_") or re.search(_CHAIN_BONE, name):
        return "bone_spine", None
    for pat, seg in _BONE_SEG:
        if re.search(pat, name):
            return "bone", seg
    s = side_of(name)
    if name.startswith("humerus_") and s:
        return "bone", f"uparm_{s}"
    if name.startswith("femur_") and s:
        return "bone", f"thigh_{s}"
    return "soft", None


VERT_ORDER = ["L5", "L4", "L3", "L2", "L1"] + [f"T{k}" for k in range(12, 0, -1)] + [f"C{k}" for k in range(7, 0, -1)]


def chain_names(centroids):
    """Present vertebrae ordered bottom -> top: the kinematic spine chain."""
    return [f"vertebrae_{x}" for x in VERT_ORDER if f"vertebrae_{x}" in centroids]


def attach_for(name, chain):
    """Bones that ride on specific vertebrae (costovertebral / sternal / girdle attachment).
    -> [(chain_index, weight)] or None."""
    import re
    idx = {n: i for i, n in enumerate(chain)}
    m = re.match(r"^rib_(?:left|right)_(\d+)$", name)
    if m:
        want = [f"vertebrae_T{m.group(1)}"]
    elif name == "sternum":
        want = [f"vertebrae_T{k}" for k in range(2, 8)]
    elif name == "costal_cartilages":
        want = [f"vertebrae_T{k}" for k in range(5, 10)]
    elif name.startswith("clavicula_"):
        want = ["vertebrae_T1", "vertebrae_T2"]
    elif name.startswith("scapula_"):
        want = [f"vertebrae_T{k}" for k in range(3, 7)]
    else:
        return None
    want = [w for w in want if w in idx]
    return [(idx[w], 1.0 / len(want)) for w in want] if want else None


def frame_layout(chain):
    """Frame order used for soft-tissue skinning: pelvis segment, vertebrae bottom->top, head
    segment, then the four limb segments."""
    return ["seg:pelvis"] + chain + ["seg:head"] + [f"seg:{n}" for n in ("uparm_l", "uparm_r", "thigh_l", "thigh_r")]


def soft_frame_weights(name, verts, lv, chain, chain_y, k=3):
    """Per-vertex influences over frame_layout(chain): the two chain frames bracketing the vertex
    height (linear), plus the thigh frame for gluteal/psoas muscles below the hip joint."""
    layout = frame_layout(chain)
    ys = [lv["pelvis"]] + list(chain_y) + [chain_y[-1] - 0.045 if len(chain_y) else lv["C2"] - 0.03]
    n_chain = len(ys)                       # pelvis + vertebrae + head
    ys = np.array(ys)                       # strictly decreasing (higher = more negative y)
    y = verts[:, 1]
    hi = np.clip(np.searchsorted(-ys, -y, side="left"), 1, n_chain - 1)   # index of frame above (smaller y)
    lo = hi - 1
    t = np.clip((ys[lo] - y) / np.maximum(ys[lo] - ys[hi], 1e-6), 0, 1)
    W = np.zeros((len(verts), len(layout)))
    r = np.arange(len(verts))
    W[r, lo] = 1 - t
    W[r, hi] = t
    sd = side_of(name)
    if sd and name.startswith(("gluteus", "iliopsoas")):
        wt = _smooth((y - (lv["hip_y"] - 0.03)) / 0.12)
        W *= (1 - wt)[:, None]
        W[:, layout.index(f"seg:thigh_{sd}")] = wt
    order = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, order, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return order.astype(np.uint8), w.astype(np.float32)


def bone_weights(name, centroid, lv):
    kind, seg = classify(name)
    if kind == "bone":
        return [(SEG[seg], 1.0)]
    w = chain_weights_y(np.array([centroid[1]]), lv)[0]
    return [(SEG[CHAIN[i]], float(w[i])) for i in range(len(CHAIN)) if w[i] > 1e-3]


def soft_weights(name, verts, lv, k=3):
    """Per-vertex LBS influences (idx (n,k) uint8, w (n,k) float32) for soft tissue."""
    W = np.zeros((len(verts), len(SEGMENTS)))
    W[:, :len(CHAIN)] = chain_weights_y(verts[:, 1], lv)
    s = side_of(name)
    if s and name.startswith(("gluteus", "iliopsoas")):          # attach to the thigh below the hip joint
        wthigh = _smooth((verts[:, 1] - (lv["hip_y"] - 0.03)) / 0.12)
        W = W * (1 - wthigh)[:, None]
        W[:, SEG[f"thigh_{s}"]] = wthigh
    order = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, order, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return order.astype(np.uint8), w.astype(np.float32)


def normalise_limb(name, verts, joints, head_center_fn):
    """Swing a CT limb bone about its proximal joint so its axis matches the MHR limb axis."""
    s = side_of(name)
    if not s:
        return verts
    if name.startswith("humerus_"):
        a, b = joints[KP[f"{s}Sh"]], joints[KP[f"{s}El"]]
    elif name.startswith("femur_"):
        a, b = joints[KP[f"{s}Hip"]], joints[KP[f"{s}Kn"]]
    else:
        return verts
    pivot, _ = head_center_fn(verts)
    c = verts.mean(0)
    axis = c - pivot
    if np.linalg.norm(axis) < 1e-6:
        return verts
    axis /= np.linalg.norm(axis)
    tgt = (b - a) / np.linalg.norm(b - a)
    R = _rot_between(axis, tgt)
    return (verts - pivot) @ R.T + pivot


def _rot_between(a, b):
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        p = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


# ------------------------------------------------------------------ articulated groups and joints
def bone_group(name):
    """Bones that articulate with each other keep the CT subject's own relative geometry: each group is
    placed with ONE rigid transform, so heads sit in sockets, ribs on vertebrae, skull on C1 by construction."""
    import re
    if name == "skull" or re.match(r"^vertebrae_C\d$", name):
        return "head_neck"
    if re.match(r"^vertebrae_T\d+$", name) or re.match(r"^(rib_|sternum|costal_cartilages|clavicula_|scapula_|humerus_)", name):
        return "thorax"
    if re.match(r"^vertebrae_L\d$", name):
        return "lumbar"
    if re.match(r"^(sacrum|hip_|femur_|vertebrae_S1)", name):
        return "pelvis"
    return None


# max relative rotation (degrees) of the child w.r.t. its parent frame. Isotropic cones, not anatomical envelopes:
# they only stop skin noise from producing impossible poses. Literature ranges in comments.
JOINT_LIMIT_DEG = {
    "glenohumeral": 160.0,      # elevation ~180, rotation ~ 90+
    "sternoclavicular": 40.0,   # elevation ~45, protraction ~ 30
    "acromioclavicular": 35.0,  # scapula on clavicle ~ 30 combined
    "hip": 120.0,               # flexion ~120
    "atlanto_occipital": 75.0,  # skull on the cervical spine: flexion/extension 60, rotation 80 (incl. C1-C2)
    "costovertebral": 12.0,     # respiratory rib rotation
}


def limit_rotation(Rrel, max_deg):
    """Clamp the angle of a relative rotation to max_deg (axis kept)."""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(Rrel).as_rotvec()
    ang = np.linalg.norm(rv)
    m = np.radians(max_deg)
    return Rotation.from_rotvec(rv * (m / ang)).as_matrix() if ang > m else Rrel


# quiet-breathing rib rotation about the costovertebral neck axis at full inspiration (degrees, depth = 1).
# Upper ribs move like a pump handle, lower ribs like a bucket handle; 11-12 barely move. Calibrated against
# tidal volume in tests/validate_breathing.py, not taken as ground truth.
RESP_DEG = {1: 2.0, 2: 2.5, 3: 3.0, 4: 3.5, 5: 4.0, 6: 4.0, 7: 4.0, 8: 4.0, 9: 3.5, 10: 3.0, 11: 1.5, 12: 1.0}
