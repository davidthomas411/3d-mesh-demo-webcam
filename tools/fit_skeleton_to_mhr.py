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
  2 SKULL   Rotation, translation and three scales (width, height, depth) so the outer surface of the skull lies a scalp thickness
            under the head skin. The brain moves with it.
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


def fit_trunk(atlas, depth):
    mesh = depth.mesh
    t = chest_table(atlas, mesh, THORAX_LEVELS)
    report_chest("before", t)
    skin_w, rib_w = t[:, 2] - t[:, 1], t[:, 6] - t[:, 5]
    skin_d, bone_d = t[:, 4] - t[:, 3], t[:, 8] - t[:, 7]
    # the largest scale that keeps the chosen tissue at the tightest level (10th percentile, so one odd level does not decide)
    sx = float(np.clip(np.percentile((skin_w - 2 * TISSUE["thorax_lateral"]) / rib_w, 10), 1.0, 1.3))
    back = t[:, 4] - t[:, 8]                                   # the posterior line is kept, so the back tissue stays as it is
    sz = float(np.clip(np.percentile((skin_d - TISSUE["thorax_anterior"] - np.maximum(back, TISSUE["thorax_posterior"])) / bone_d, 10), 1.0, 1.3))
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
    print(f"  thorax scaled x{sx:.3f} side to side and x{sz:.3f} front to back, shifted {(xm_new - xm) * 1000:+.0f} mm sideways; pelvis and neck unchanged")
    return {"lateral": round(sx, 4), "anteroposterior": round(sz, 4), "shift_mm": round((xm_new - xm) * 1000, 1)}


def fit_skull(atlas, depth):
    J = {j["name"]: j for j in atlas.meta["joints"]}
    Jao = np.asarray(J["ao"]["center"], np.float64)
    skull = atlas.get("skull")
    pts = sample(skull, 700)
    target = TISSUE["scalp"]

    def tf(x, p):                                              # about the atlanto-occipital centre
        return Jao + x[6:9] + ((p - Jao) * np.exp(x[3:6])) @ Rot.from_rotvec(x[0:3]).as_matrix().T

    def residuals(x):
        d = depth(tf(x, pts), exact=True)
        near = d < 0.022                                       # the outer surface: vault and face (the skull base is deep inside)
        e = np.where(near, d - target, 0.0) * 1000
        e = np.sign(e) * np.minimum(np.abs(e), 6 + 0.25 * np.abs(e))          # soft cap: the face is not a constant thickness
        return np.concatenate([e / np.sqrt(len(pts)) * 2.0,
                               hinge(d, 0.003) * 1000 / np.sqrt(len(pts)) * 6.0,   # never closer than 3 mm to the skin
                               np.degrees(x[0:3]) * 0.04,
                               (x[3:6] - x[3:6].mean()) * 100 * 0.06,            # the three scales stay close to each other
                               x[6:9] * 1000 * 0.08])                            # the skull base stays on the neck

    before = outside_report(depth, skull)
    d0 = depth(pts, exact=True)
    sol = least_squares(residuals, np.zeros(9), method="trf", x_scale=0.02, diff_step=2e-3, max_nfev=80)
    x = sol.x
    for name, e in atlas.by.items():
        if name == "skull" or (e.get("bind") == "frame" and e.get("frame") == "skull"):
            atlas.put(name, tf(x, atlas.get(name)))
    J["ao"]["center"] = [round(float(v), 5) for v in tf(x, Jao[None])[0]]
    after = outside_report(depth, atlas.get("skull"))
    d1 = depth(tf(x, pts), exact=True)
    s = np.exp(x[3:6])
    near0, near1 = d0[d0 < 0.022] * 1000, d1[d1 < 0.022] * 1000
    print(f"  skull: width x{s[0]:.3f}, height x{s[1]:.3f}, depth x{s[2]:.3f}, turned {np.degrees(np.linalg.norm(x[0:3])):.1f} deg, moved {np.linalg.norm(x[6:9]) * 1000:.1f} mm")
    print(f"  outside the head skin {before[0]:.1f}% -> {after[0]:.1f}% | outer surface depth under the skin: median {np.median(near0):.1f} -> {np.median(near1):.1f} mm, "
          f"spread (10-90%) {np.percentile(near0, 10):.1f}..{np.percentile(near0, 90):.1f} -> {np.percentile(near1, 10):.1f}..{np.percentile(near1, 90):.1f} mm")
    return {"scale_xyz": [round(float(v), 4) for v in s], "rotation_deg": round(float(np.degrees(np.linalg.norm(x[0:3]))), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="CT-space atlas prefix (atlas/ct_subject)")
    ap.add_argument("out", help="fitted atlas prefix (atlas/atlas)")
    ap.add_argument("rest", help="mhr_rest.npz: neutral MHR mesh (verts, faces)")
    ap.add_argument("mhr", help="mhr_model.pt")
    a = ap.parse_args()
    for ext in (".json", ".bin"):
        shutil.copyfile(a.source + ext, a.out + ext)
    rest = np.load(a.rest)
    depth = SkinDepth(np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3))
    atlas = AtlasFile(a.out)
    print("1 TRUNK")
    trunk = fit_trunk(atlas, depth)
    print("2 SKULL")
    skull = fit_skull(atlas, depth)
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
