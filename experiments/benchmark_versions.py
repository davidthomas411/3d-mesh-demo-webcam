"""BENCHMARK: the same real CT and the same body motion, run through each stage of the project's anatomical realism.

Stages (each adds one thing; all are numpy and reproducible from this repo):
  V0  skin-driven soft warp        no skeleton: the surface drives everything, bones smear
  V1  per-structure rigid fits     each structure chases its own skin patch (the first rig; incoherent skeleton)
  V2  rigid segments + spine chain coherent skeleton: segments from skin, vertebrae as a kinematic chain, ribs on vertebrae, no joints
  V3  + joint rules                pinned ball/saddle/condyloid joints, clavicle aim, scapulohumeral rhythm, range-of-motion limits
  V4  + in-vivo spine distribution trunk rotation shared along the vertebrae by measured per-level data (Fujimori 2012 CT; lumbar CT/radiography)
Planned next (not implemented): breathing (rib/diaphragm/lung), scapulothoracic contact, soft-tissue sliding, costal-cartilage flexibility,
  coupled bending-rotation, natural motion timing and asymmetry.

usage:
  python experiments/benchmark_versions.py metrics OUT.json
  python experiments/benchmark_versions.py render OUTDIR [--workers 3] [--frames-per-phase 12,9,9]
env: SUBJECT, CORR, ATLAS, ONNX, REST (paths) as in tools/render_radiology.py
"""
import os, sys, json
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from anatomy.runtime import Atlas, kabsch, _AXIAL

VERSIONS = [
    ("V0", "Skin-driven warp", "no skeleton: the body surface\ndrives everything; bones smear"),
    ("V1", "Per-structure fits", "each bone chases its own skin\npatch: incoherent skeleton"),
    ("V2", "Rigid segments + chain", "coherent skeleton, spine as a\nkinematic chain; no joints"),
    ("V3", "+ Joint rules", "pinned shoulder/hip/skull joints,\nclavicle aim, range-of-motion"),
    ("V4", "+ In-vivo spine split", "trunk rotation shared along\nvertebrae by measured data"),
]


class PerStructure:
    """V1: every structure is moved rigidly by the fit of its own nearest skin patch (the project's first rig)."""
    def __init__(self, atlas, rest_skin):
        tree = cKDTree(rest_skin)
        self.atlas, self.sup = atlas, []
        for s in atlas.structures:
            pick = np.random.default_rng(0).choice(len(s["pos"]), min(600, len(s["pos"])), replace=False)
            _, nn = tree.query(s["pos"][pick], k=3)
            ids, cnt = np.unique(nn.ravel(), return_counts=True)
            o = np.argsort(-cnt)[:192]
            self.sup.append((ids[o], cnt[o].astype(float), rest_skin[ids[o]]))

    def pose(self, skin):
        out = {}
        for s, (ids, w, rest) in zip(self.atlas.structures, self.sup):
            R, cs, cd = kabsch(rest, skin[ids], w)
            out[s["name"]] = (s["pos"] - cs) @ R.T + cd
        return out


def posed_for(version, atlas, skin, ps):
    if version == "V0":
        return None
    if version == "V1":
        return ps.pose(skin)
    if version == "V2":
        return atlas.pose(skin, joints=False)
    if version == "V3":
        return atlas.pose(skin)
    return atlas.pose(skin, spine="physio")


# ------------------------------------------------------------------------------------------------ motion script
def motion_params(phase, u):
    import render_radiology as rr
    p = np.zeros((1, 204), np.float32)
    for i, v in rr.PHASES[phase][1](u).items():
        p[0, i] = v
    return rr.PHASES[phase][0], p


# ------------------------------------------------------------------------------------------------ numeric scores
def joint_gap(atlas, j, posed):
    by = {s["name"]: s for s in atlas.structures}
    J = np.asarray(j["center"]); c = by[j["child"]]["pos"]
    sel = np.linalg.norm(c - J, axis=1) < 0.035
    if sel.sum() < 5:
        sel = np.argsort(np.linalg.norm(c - J, axis=1))[:30]
    return float(cKDTree(posed[j["parent"]]).query(posed[j["child"]][sel])[0].min())


def head_in_socket_change(atlas, posed, side):
    """Change (mm) of the humeral-head-centre to scapula-surface distance, rest -> posed: how far the ball leaves/enters its socket.
    (A minimum-surface-gap score misses overlap, which also has a gap near zero.)"""
    from anatomy.atlas_build import joint_head_center
    by = {s["name"]: s["pos"] for s in atlas.structures}
    hc = lambda X: joint_head_center(X * 1000.0, -1)[0] / 1000.0
    d0 = cKDTree(by[f"scapula_{side}"]).query(hc(by[f"humerus_{side}"]))[0]
    d1 = cKDTree(posed[f"scapula_{side}"]).query(hc(posed[f"humerus_{side}"]))[0]
    return float(1000 * (d1 - d0))


def spine_scores(atlas, posed):
    """Axial (about the vertical axis) rotation of every lumbar+thoracic vertebra, then the per-level increments."""
    by = {s["name"]: s for s in atlas.structures}
    names = [n for n in atlas.meta["chain"] if n.replace("vertebrae_", "") in _AXIAL]            # L5 .. T1, bottom to top
    psi = np.array([np.degrees(Rot.from_matrix(kabsch(by[n]["pos"], posed[n])[0]).as_rotvec()[1]) for n in names])
    inc = np.diff(psi)                                                                         # L4-L5 ... T1-T2 (16 increments)
    total = psi[-1] - psi[0]
    inc = inc * (np.sign(total) if abs(total) > 1e-9 else 1.0)                                 # compare in the direction of the twist
    ref = np.array([_AXIAL[n.replace("vertebrae_", "")] for n in names[1:]])
    r = float(np.corrcoef(inc, ref)[0, 1]) if np.std(inc) > 1e-9 else float("nan")
    lum = psi[names.index("vertebrae_L1")] - psi[0]
    return {"total_deg": float(total), "lumbar_share": float(lum / total) if abs(total) > 1e-9 else float("nan"), "profile_r": r,
            "per_level_deg": [round(float(x), 2) for x in inc]}


def spacing_change(atlas, posed):
    by = {s["name"]: s for s in atlas.structures}
    ch = atlas.meta["chain"]
    cen = lambda P, n: P[n].mean(0)
    rest = {n: by[n]["pos"] for n in ch}
    return 1000 * max(abs(np.linalg.norm(cen(posed, a) - cen(posed, b)) - np.linalg.norm(cen(rest, a) - cen(rest, b))) for a, b in zip(ch[:-1], ch[1:]))


def metrics(out_json):
    import onnxruntime as ort
    atlas = Atlas(os.environ["ATLAS"])
    rest = np.load(os.environ["REST"])["verts"].astype(float)
    sess = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    ps = PerStructure(atlas, rest)
    skin_of = lambda p: sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    peaks = {"twist": (0, 0.25), "shrug": (2, 0.5), "forward bend": (4, 0.5)}
    skins = {k: skin_of(motion_params(ph, u)[1]) for k, (ph, u) in peaks.items()}
    joints = [j for j in atlas.meta["joints"] if j["type"] != "costovertebral"]
    rest_pos = {s["name"]: s["pos"] for s in atlas.structures}
    res = {}
    for v, name, _ in VERSIONS[1:]:
        row = {"name": name}
        posed_s = posed_for(v, atlas, skins["shrug"], ps)
        row["joint_gap_change_mm_under_shrug"] = {j["name"]: round(1000 * abs(joint_gap(atlas, j, posed_s) - joint_gap(atlas, j, rest_pos)), 1) for j in joints}
        row["worst_joint_gap_change_mm"] = max(row["joint_gap_change_mm_under_shrug"].values())
        row["head_in_socket_change_mm"] = {sd: round(head_in_socket_change(atlas, posed_s, sd), 1) for sd in ("left", "right")}
        row["glenohumeral_gap_change_mm"] = max(abs(x) for x in row["head_in_socket_change_mm"].values())
        row["twist"] = spine_scores(atlas, posed_for(v, atlas, skins["twist"], ps))
        row["max_vertebra_spacing_change_mm"] = round(max(spacing_change(atlas, posed_for(v, atlas, skins[k], ps)) for k in skins), 1)
        res[v] = row
    json.dump(res, open(out_json, "w"), indent=1)
    ref_total = sum(_AXIAL.values())
    print(f"reference (in-vivo CT, per-level increments): lumbar share of axial rotation {sum(_AXIAL[k] for k in ('L5','L4','L3','L2','L1'))/ref_total:.0%}")
    print(f"{'stage':28s} {'head-in-socket':>15s} {'worst other joint':>18s} {'spacing chg':>12s} {'lumbar share':>13s} {'profile r':>10s}")
    for v, r in res.items():
        print(f"{v + ' ' + r['name']:28s} {r['glenohumeral_gap_change_mm']:12.1f} mm {r['worst_joint_gap_change_mm']:15.1f} mm {r['max_vertebra_spacing_change_mm']:9.1f} mm"
              f" {r['twist']['lumbar_share']:12.0%} {r['twist']['profile_r']:10.2f}")
    return res


# ------------------------------------------------------------------------------------------------ rendering
W = CFG = SESS = PS = REST = None


def setup():
    global W, CFG, SESS, PS, REST
    import onnxruntime as ort
    from ct_warp import CTWarper
    REST = np.load(os.environ["REST"])["verts"].astype(float)
    SESS = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    W = CTWarper(os.environ["SUBJECT"], os.environ["CORR"], os.environ["ATLAS"], REST)
    PS = PerStructure(W.atlas, REST)
    spine = np.vstack([W.v_ct[n] for n in W.names if n.startswith("vertebrae_")])
    skull = W.v_ct["skull"]; pelvis = np.vstack([W.v_ct[n] for n in W.names if n.startswith(("hip_", "sacrum"))])
    CFG = {"c": spine.mean(0), "y": (skull[:, 1].min() - 60, pelvis[:, 1].max() + 60)}


def panel(version, posed, skin, cor_z, centre):
    import render_radiology as rr
    c, (y0, y1) = CFG["c"], CFG["y"]
    fld = W.field(posed, skin, structures=(version != "V0")) if posed is not None else W.field({}, skin, structures=False)
    region = (np.array([c[0] - 260, y0, c[2] - 330.0]), np.array([c[0] + 260, y1, c[2] + 220.0]))
    vol, bone, axes, st = W.warped_volume(fld, step=3.0, region=region)
    L = rr.radiograph(vol, bone, 2, st)                                    # bone-only AP radiograph
    u8 = rr.window(L.T, 0.0, 20.0, 0.75)
    u8 = rr.unsharp(u8, 2.2, 1.0)
    ap = Image.fromarray(u8).convert("RGB").resize((330, int(u8.shape[0] * 330 / u8.shape[1])), Image.BICUBIC)
    P, shape = W.plane(2, cor_z, 0, 1, centre, 260.0, (y1 - y0) / 2, 2.0)  # coronal reformat that follows the patient (through T9)
    src = W.source_positions(fld, P)
    hu = W.sample(W.hu_c, src).reshape(shape).astype(np.float32)
    g = rr.unsharp(rr.window(hu, 40 - 175, 40 + 175), 1.4, 0.6)
    rgb = np.stack([g] * 3, -1)                                             # real HU, bone not masked
    cor = Image.fromarray(rgb).resize((330, int(rgb.shape[0] * 330 / rgb.shape[1])), Image.LANCZOS)
    return ap, cor


def job(args):
    k, ph, u = args
    import render_radiology as rr
    name, p = motion_params(ph, u)
    skin = SESS.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed3 = W.atlas.pose(skin)
    s_, R_, t_ = W.T["vertebrae_T9"]
    c9 = (((posed3["vertebrae_T9"] - t_) @ R_) / s_).mean(0)               # posed T9 in the CT frame
    cor_z = c9[2] - 55.0
    centre = np.array([CFG["c"][0], (CFG["y"][0] + CFG["y"][1]) / 2, CFG["c"][2]])
    cols = [panel(v, posed_for(v, W.atlas, skin, PS), skin, cor_z, centre) for v, _, _ in VERSIONS]
    hs = [a.height + c.height for a, c in cols]
    head, foot, gap = 110, 44, 6
    canvas = Image.new("RGB", (330 * 5 + gap * 4, max(hs) + head + foot), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "Anatomical-realism benchmark: the same real CT, the same motion, five stages of the model", fill=(240, 240, 240), font=rr.font(22))
    d.text((10, 36), f"motion: {name}   |   top: bone-only AP radiograph   bottom: coronal soft-tissue reformat through T9 (real HU, W350/L40)", fill=(170, 200, 230), font=rr.font(14))
    for i, ((v, title, cap), (ap, cor)) in enumerate(zip(VERSIONS, cols)):
        x = i * (330 + gap)
        d.text((x + 4, 60), f"{v}  {title}", fill=(255, 220, 120), font=rr.font(16))
        d.text((x + 4, 82), cap.replace("\n", " "), fill=(150, 160, 175), font=rr.font(11))
        canvas.paste(ap, (x, head)); canvas.paste(cor, (x, head + ap.height))
    d.text((10, canvas.height - 38), "Synthetic: the atlas subject's real CT deformed by each model stage. Not patient care data. No breathing, scapulothoracic contact or soft-tissue sliding yet.", fill=(225, 190, 90), font=rr.font(12))
    d.text((10, canvas.height - 20), "Stages V0-V4 use identical input; V4's spine split uses in-vivo CT data (Fujimori 2012) and assumed lumbar values; see docs/BENCHMARK.md.", fill=(130, 130, 130), font=rr.font(11))
    canvas.save(os.path.join(OUT, f"{k:03d}.png"))
    return k


OUT = "."
if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "metrics":
        metrics(a[1])
    else:
        OUT = a[1]; os.makedirs(OUT, exist_ok=True)
        per = [int(x) for x in a[a.index("--frames-per-phase") + 1].split(",")] if "--frames-per-phase" in a else [12, 9, 9]
        workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 3
        plan = []
        for ph, n in zip((0, 2, 4), per):                                  # twist, shoulder shrug, forward bend
            plan += [(len(plan) + i, ph, i / n) for i in range(n)]
        setup()
        with Pool(workers) as pool:
            for done in pool.imap_unordered(job, plan):
                print("frame", done, "of", len(plan), flush=True)
        print("finished")
