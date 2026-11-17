"""Fit the CT subject's skeleton (and the organs it carries) to the MHR body: ONE definition of the rest pose, run from the
CT-space atlas, instead of separate fix-ups per bone.

Why: the atlas is one person's CT registered into the neutral MHR body. Registration put it in the right place but left it the CT
subject's size and proportions, which are not the model's. Measured against the neutral body before this fit: rib cage 262 mm wide in
a 316-358 mm chest with 44-62 mm of tissue in front of it and 10 mm off-centre; humeral heads 63-88 mm from the MHR shoulder joints;
7% of the skull outside the head.

The rules, applied in this order to atlas/ct_subject.* (the atlas as built by anatomy.atlas_build):

  1 TRUNK   One in-plane scale for the thorax (lateral and front-back separately), the largest that still leaves a chosen thickness
            of soft tissue between the ribs and the skin at the tightest level, about the spine's posterior line, and a lateral shift
            that centres the rib cage in the chest. It is a smooth warp of space, so EVERYTHING in the thorax moves together: ribs,
            sternum, vertebrae, heart, lungs, vessels. It fades to nothing through the neck and through the abdomen; the pelvis keeps
            its size.
  2 SKULL   Rotation, translation and three scales (width, height, depth) so the skull fills the head: its vault a scalp thickness
            under the head skin, its orbits at the height of the model's eye joints. The brain moves with it, the neck follows.
  3 JOINTS  Rig segments are bound to MHR joints (tools/bind_mhr_joints.py).
  4 ARMS    Humeral head on the MHR shoulder joint, shaft on the shoulder-elbow line, girdle sized to meet it
            (tools/seat_limbs_on_mhr.py).

What is assumed, not measured: the soft-tissue thicknesses below. They are typical lean-adult values; the neutral MHR body has no
anatomy inside it to check them against. The hips are not re-seated (the atlas hip centres are 24-26 mm from the MHR hip joints).

usage: python tools/fit_skeleton_to_mhr.py atlas/ct_subject atlas/atlas mhr_rest.npz mhr_model.pt"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from reseat_bones import SkinDepth, hinge, outside_report, sample  # noqa: E402

# soft tissue between bone and skin that the fit leaves (metres)
TISSUE = {"thorax_lateral": 0.020, "thorax_anterior": 0.025, "thorax_posterior": 0.030, "scalp": 0.007}
THORAX_LEVELS = np.arange(-1.30, -1.11, 0.02)     # rest-frame y (down) where the rib cage is measured against the chest
NECK_FADE = (-1.50, -1.43)                         # the warp is zero above the first value, full below the second
ABDOMEN_FADE = (-1.10, -1.00)                      # full above the first value, zero below the second (pelvis untouched)


def smoothstep(e0, e1, y):
    t = np.clip((y - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def blend(y):
    return smoothstep(*NECK_FADE, y) * (1.0 - smoothstep(*ABDOMEN_FADE, y))


class AtlasFile:
    def __init__(self, prefix):
        self.prefix = prefix
        self.meta = json.load(open(prefix + ".json"))
        self.blob = bytearray(open(prefix + ".bin", "rb").read())
        self.by = {e["name"]: e for e in self.meta["structures"]}

    def get(self, name):
        e = self.by[name]
        return np.frombuffer(self.blob, np.float32, e["n_verts"] * 3, e["pos"]).reshape(-1, 3).astype(np.float64)

    def put(self, name, pos):
        e = self.by[name]
        self.blob[e["pos"]:e["pos"] + e["n_verts"] * 12] = pos.astype("<f4").tobytes()
        e["centroid"] = [round(float(v), 5) for v in pos.mean(0)]

    def save(self, note):
        self.meta.setdefault("history", []).append(note)
        json.dump(self.meta, open(self.prefix + ".json", "w"), separators=(",", ":"))
        open(self.prefix + ".bin", "wb").write(bytes(self.blob))


def chest_table(atlas, mesh, levels):
    """Per level: skin (x_min, x_max, z_min, z_max) of the torso section and the same for the thoracic bones in a 2 cm slab."""
    ribs = np.vstack([atlas.get(n) for n in atlas.by if n.startswith("rib_")])
    bones = np.vstack([atlas.get(n) for n, e in atlas.by.items() if e["group"] in ("ribcage", "spine")])
    rows = []
    for y in levels:
        sec = mesh.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        if sec is None:
            continue
        p = sec.vertices
        p = p[np.abs(p[:, 0]) < 0.22]                         # the torso, not the arms
        r, b = ribs[np.abs(ribs[:, 1] - y) < 0.01], bones[np.abs(bones[:, 1] - y) < 0.01]
        if len(r) < 20 or len(p) < 20:
            continue
        rows.append((y, p[:, 0].min(), p[:, 0].max(), p[:, 2].min(), p[:, 2].max(), r[:, 0].min(), r[:, 0].max(), b[:, 2].min(), b[:, 2].max()))
    return np.array(rows)


def report_chest(tag, t):
    lat = np.minimum(t[:, 5] - t[:, 1], t[:, 2] - t[:, 6]) * 1000
    print(f"  {tag}: rib cage {np.max(t[:, 6] - t[:, 5]) * 1000:.0f} mm wide, {np.max(t[:, 8] - t[:, 7]) * 1000:.0f} mm deep | tissue to the skin: "
          f"side {lat.min():.0f}-{lat.max():.0f} mm (left/right differ by {np.abs((t[:, 5] - t[:, 1]) - (t[:, 2] - t[:, 6])).mean() * 1000:.0f} mm), "
          f"front {((t[:, 7] - t[:, 3]) * 1000).min():.0f}-{((t[:, 7] - t[:, 3]) * 1000).max():.0f} mm, back {((t[:, 4] - t[:, 8]) * 1000).min():.0f}-{((t[:, 4] - t[:, 8]) * 1000).max():.0f} mm")


def skel_thorax_size(skel_fit):
    """Width and depth (m) of SKEL's rib cage in its fit to this body (tools/fit_skel_to_mhr.py)."""
    z = np.load(skel_fit, allow_pickle=True)
    names = [str(n) for n in z["bone_names"]]
    a, e = z["bone_verts_indices"][names.index("thorax")]
    t = z["skel_verts"][a:e]
    return float(np.ptp(t[:, 0])), float(np.ptp(t[:, 2]))


def fit_trunk(atlas, depth, skel_fit=None):
    mesh = depth.mesh
    t = chest_table(atlas, mesh, THORAX_LEVELS)
    report_chest("before", t)
    skin_w, rib_w = t[:, 2] - t[:, 1], t[:, 6] - t[:, 5]
    skin_d, bone_d = t[:, 4] - t[:, 3], t[:, 8] - t[:, 7]
    # the largest scale that keeps the chosen tissue at the tightest level (10th percentile, so one odd level does not decide)
    sx = float(np.clip(np.percentile((skin_w - 2 * TISSUE["thorax_lateral"]) / rib_w, 10), 1.0, 1.3))
    back = t[:, 4] - t[:, 8]                                   # the posterior line is kept, so the back tissue stays as it is
    sz = float(np.clip(np.percentile((skin_d - TISSUE["thorax_anterior"] - np.maximum(back, TISSUE["thorax_posterior"])) / bone_d, 10), 1.0, 1.3))
    source = "assumed soft-tissue thickness"
    if skel_fit:
        # a learned reference is available: size the rib cage as SKEL's is in this body, instead of from assumed tissue thickness
        ribs_all = np.vstack([atlas.get(n) for n in atlas.by if n.startswith("rib_")])
        w, d = skel_thorax_size(skel_fit)
        sx = float(np.clip(w / np.ptp(ribs_all[:, 0]), 0.85, 1.25))
        sz = float(np.clip(d / np.ptp(ribs_all[:, 2]), 0.85, 1.25))
        source = f"SKEL rib cage {w * 1000:.0f} x {d * 1000:.0f} mm"
    xm = float(np.median((t[:, 5] + t[:, 6]) / 2))             # rib-cage midline now
    xm_new = float(np.median((t[:, 1] + t[:, 2]) / 2))         # chest midline
    # posterior line of the bones (most posterior point per level), smooth in y; scaling front-back is done about it
    grid = np.arange(-1.60, -0.90, 0.01)
    bones = np.vstack([atlas.get(n) for n, e in atlas.by.items() if e["group"] in ("ribcage", "spine")])
    zb = np.array([bones[np.abs(bones[:, 1] - y) < 0.012][:, 2].max() if (np.abs(bones[:, 1] - y) < 0.012).sum() > 5 else np.nan for y in grid])
    ok = ~np.isnan(zb)
    zb = gaussian_filter1d(np.interp(grid, grid[ok], zb[ok]), 3.0)

    def warp(p):
        a = blend(p[:, 1])
        z_ref = np.interp(p[:, 1], grid, zb)
        out = p.copy()
        out[:, 0] = p[:, 0] + a * ((xm_new + sx * (p[:, 0] - xm)) - p[:, 0])
        out[:, 2] = p[:, 2] + a * ((z_ref - sz * (z_ref - p[:, 2])) - p[:, 2])
        return out

    for name in atlas.by:
        atlas.put(name, warp(atlas.get(name)))
    for j in atlas.meta["joints"]:
        c = np.asarray(j["center"], np.float64)[None]
        a = float(blend(c[:, 1])[0])
        k = np.array([1 + a * (sx - 1), 1.0, 1 + a * (sz - 1)])
        j["center"] = [round(float(v), 5) for v in warp(c)[0]]
        if "axis" in j:
            ax = np.asarray(j["axis"]) * k
            j["axis"] = [round(float(v), 5) for v in ax / np.linalg.norm(ax)]
        if "ellipsoid" in j:                                   # the rib-cage ellipsoid the scapula glides on
            e = j["ellipsoid"]; ec = np.asarray(e["center"], np.float64)[None]
            ae = float(blend(ec[:, 1])[0])
            e["center"] = [round(float(v), 5) for v in warp(ec)[0]]
            e["radii"] = [round(float(v), 5) for v in np.asarray(e["radii"]) * np.array([1 + ae * (sx - 1), 1.0, 1 + ae * (sz - 1)])]
    report_chest("after ", chest_table(atlas, mesh, THORAX_LEVELS))
    print(f"  thorax scaled x{sx:.3f} side to side and x{sz:.3f} front to back ({source}), shifted {(xm_new - xm) * 1000:+.0f} mm sideways; pelvis and neck unchanged")
    return {"lateral": round(sx, 4), "anteroposterior": round(sz, 4), "shift_mm": round((xm_new - xm) * 1000, 1), "size_from": source}


def head_landmarks(mhr_path):
    """Eye joints, head-top joint and head joint of the neutral MHR skeleton (atlas frame), found from the hierarchy:
    the children of the head joint are a mirrored pair (the eyes), a joint straight above (the top of the head) and the jaw."""
    from bind_mhr_joints import MHR
    m = MHR(mhr_path)
    _, p0, _ = m.pose()
    w = np.bincount(m.w_joint, weights=m.w, minlength=len(m.parents))
    head = max(range(len(m.parents)), key=lambda k: w[k] if p0[k][1] < -1.5 else -1)     # most skin weight above the shoulders
    kids = m.children(head)
    eyes = sorted([k for k in kids if abs(p0[k][0]) > 0.01], key=lambda k: p0[k][0])
    top = min(kids, key=lambda k: p0[k][1])
    return {"eye_r": p0[eyes[0]], "eye_l": p0[eyes[-1]], "top": p0[top], "head": p0[head]}


def skull_orbit_height(skull, faces):
    """Height (y) of the orbits on the skull: seen from the front, the eye sockets are where the bone surface is deepest
    between the forehead and the cheekbones."""
    import trimesh
    from scipy.ndimage import gaussian_filter1d as g1
    sm = trimesh.Trimesh(skull, faces, process=False)
    top, h = skull[:, 1].min(), np.ptp(skull[:, 1])
    ys = np.arange(top + 0.2 * h, top + 0.65 * h, 0.002)
    xm = np.median(skull[:, 0])
    prof = []
    for y in ys:
        o = np.array([[xm + sx_ * dx, y, -0.5] for sx_ in (-1, 1) for dx in (0.026, 0.032, 0.038)])
        loc, ir, _ = sm.ray.intersects_location(o, np.tile([0, 0, 1.0], (len(o), 1)), multiple_hits=False)
        z = np.full(len(o), skull[:, 2].max()); z[ir] = loc[:, 2]
        prof.append(np.median(z))
    prof = g1(np.array(prof), 2.0)
    return float(ys[int(np.argmax(prof))]), float(xm)


def fit_skull(atlas, depth, lm):
    """The skull fills the head: its vault lies a scalp thickness under the head skin and its orbits are at the height of the
    model's eyes. Rotation, translation and three scales (width, height, depth); the brain moves with it and the neck follows."""
    import trimesh
    J = {j["name"]: j for j in atlas.meta["joints"]}
    Jao = np.asarray(J["ao"]["center"], np.float64)
    e = atlas.by["skull"]
    skull = atlas.get("skull")
    faces = np.frombuffer(atlas.blob, np.uint16 if e["idx_type"] == "u16" else np.uint32, e["n_tris"] * 3, e["idx"]).reshape(-1, 3).astype(np.int64)
    orbit_y, mid_x = skull_orbit_height(skull, faces)
    c = skull.mean(0)
    top, h = skull[:, 1].min(), np.ptp(skull[:, 1])
    # outer surface of the vault: on the convex hull, in the upper part, not facing down
    hull = trimesh.Trimesh(skull, faces, process=False).convex_hull
    on_hull = np.abs(trimesh.proximity.signed_distance(hull, skull)) < 0.003
    vault = skull[on_hull & (skull[:, 1] < top + 0.5 * h)]
    vault = sample(vault, 350)
    rest = sample(skull, 500)
    orbit = np.array([[mid_x + 0.032, orbit_y, c[2]], [mid_x - 0.032, orbit_y, c[2]]])
    eye_y = float((lm["eye_l"][1] + lm["eye_r"][1]) / 2)
    target = TISSUE["scalp"]

    def tf(x, p):
        return c + x[6:9] + ((p - c) * np.exp(x[3:6])) @ Rot.from_rotvec(x[0:3]).as_matrix().T

    def residuals(x):
        dv = depth(tf(x, vault), exact=True)
        ev = (dv - target) * 1000
        ev = np.sign(ev) * np.minimum(np.abs(ev), 8 + 0.35 * np.abs(ev))             # robust, but a far gap still pulls
        o = tf(x, orbit)
        return np.concatenate([ev / np.sqrt(len(vault)) * 3.0,
                               hinge(depth(tf(x, rest), exact=True), 0.003) * 1000 / np.sqrt(len(rest)) * 8.0,   # all of it at least 3 mm inside
                               (o[:, 1] - eye_y) * 1000 * 0.8,                         # orbits at the height of the eyes
                               [(o[:, 0].mean() - (lm["eye_l"][0] + lm["eye_r"][0]) / 2) * 1000 * 0.5],
                               np.degrees(x[0:3]) * 0.05])

    before = outside_report(depth, skull)
    gap0 = (skull[:, 1].min() - depth.mesh.vertices[:, 1].min()) * 1000
    best = None
    for start in (np.zeros(9), np.r_[0, 0, 0, 0.05, 0.25, 0.05, 0, -0.03, 0.0]):          # second start: already stretched upward
        sol = least_squares(residuals, start, method="trf", x_scale=0.03, diff_step=3e-3, max_nfev=90,
                            bounds=(np.r_[-0.6, -0.6, -0.6, -0.25, -0.25, -0.25, -0.1, -0.1, -0.1], np.r_[0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1]))
        if best is None or sol.cost < best.cost:
            best = sol
    x = best.x
    for name, en in atlas.by.items():
        if name == "skull" or (en.get("bind") == "frame" and en.get("frame") == "skull"):
            atlas.put(name, tf(x, atlas.get(name)))
    # the neck follows: whatever the skull base moved by is carried, fading out, down to the base of the neck
    ao_new = tf(x, Jao[None])[0]
    shift = ao_new - Jao

    def follow(p):
        return p + shift * smoothstep(-1.44, float(Jao[1]), -(-p[:, 1:2])) if False else p + shift * (1.0 - smoothstep(float(Jao[1]), -1.44, p[:, 1:2]))

    for name, en in atlas.by.items():
        if name == "skull" or (en.get("bind") == "frame" and en.get("frame") == "skull"):
            continue
        atlas.put(name, follow(atlas.get(name)))
    for jn in atlas.meta["joints"]:
        cc = np.asarray(jn["center"], np.float64)[None]
        jn["center"] = [round(float(v), 5) for v in (ao_new if jn["name"] == "ao" else follow(cc)[0])]
    sk2 = atlas.get("skull")
    after = outside_report(depth, sk2)
    sc = np.exp(x[3:6])
    dv = depth(tf(x, vault), exact=True) * 1000
    o = tf(x, orbit)
    print(f"  skull: width x{sc[0]:.3f}, height x{sc[1]:.3f}, depth x{sc[2]:.3f}, turned {np.degrees(np.linalg.norm(x[0:3])):.1f} deg; "
          f"now {np.ptp(sk2[:, 0]) * 1000:.0f} x {np.ptp(sk2[:, 1]) * 1000:.0f} x {np.ptp(sk2[:, 2]) * 1000:.0f} mm (w x h x d)")
    print(f"  top of the skull to top of the head: {gap0:.0f} -> {(sk2[:, 1].min() - depth.mesh.vertices[:, 1].min()) * 1000:.0f} mm | "
          f"orbits to the model's eye height: {(orbit_y - eye_y) * 1000:+.0f} -> {(o[:, 1].mean() - eye_y) * 1000:+.0f} mm (+ = too low)")
    print(f"  vault under the scalp: median {np.median(dv):.1f} mm (10-90%: {np.percentile(dv, 10):.1f}..{np.percentile(dv, 90):.1f}) | outside the head {before[0]:.1f}% -> {after[0]:.1f}% | "
          f"skull base moved {np.linalg.norm(shift) * 1000:.0f} mm, the neck follows")
    return {"scale_xyz": [round(float(v), 4) for v in sc], "rotation_deg": round(float(np.degrees(np.linalg.norm(x[0:3]))), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="CT-space atlas prefix (atlas/ct_subject)")
    ap.add_argument("out", help="fitted atlas prefix (atlas/atlas)")
    ap.add_argument("rest", help="mhr_rest.npz: neutral MHR mesh (verts, faces)")
    ap.add_argument("mhr", help="mhr_model.pt")
    ap.add_argument("--skel", default=None, help="SKEL fitted to this body (tools/fit_skel_to_mhr.py): used as the size reference for the rib cage")
    a = ap.parse_args()
    for ext in (".json", ".bin"):
        shutil.copyfile(a.source + ext, a.out + ext)
    rest = np.load(a.rest)
    depth = SkinDepth(np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3))
    atlas = AtlasFile(a.out)
    print("1 TRUNK")
    trunk = fit_trunk(atlas, depth, a.skel)
    print("2 SKULL")
    skull = fit_skull(atlas, depth, head_landmarks(a.mhr))
    atlas.meta["fit"] = {"tool": "tools/fit_skeleton_to_mhr.py", "source": os.path.basename(a.source), "tissue_m": TISSUE, "thorax": trunk, "skull": skull}
    atlas.meta.pop("history", None)
    atlas.save("tools/fit_skeleton_to_mhr.py: thorax and skull fitted to the neutral MHR body")
    print("3 JOINTS")
    subprocess.run([sys.executable, os.path.join(HERE, "bind_mhr_joints.py"), a.out, a.mhr], check=True, stdout=subprocess.DEVNULL)
    print("  rig segments bound to MHR joints")
    print("4 ARMS")
    r = subprocess.run([sys.executable, os.path.join(HERE, "seat_limbs_on_mhr.py"), a.out, a.rest], check=True, capture_output=True, text=True)
    print("  " + "\n  ".join(l for l in r.stdout.splitlines() if l.strip() and not l.startswith("wrote")))


if __name__ == "__main__":
    main()
