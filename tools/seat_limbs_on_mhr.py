"""Define the upper-arm bones on the body model's own skeleton (rest pose).

The atlas humeri came from a CT with cropped arms and were swung into place by a guess: their pivot sat 6-9 cm from the MHR shoulder
joint and their shaft 11-18 degrees off the MHR shoulder-elbow line, so the bone could not overlay the arm of the mesh it is drawn in.
SKEL / HSMR avoid this by construction (their skin is generated from the skeleton). This tool does the same thing for the arm here:

  humerus   head centre ON the MHR shoulder joint, shaft ON the shoulder->elbow line, length scaled to that joint distance
  scapula   glenoid on the same joint (so the ball is in its socket), turned and sized to stay under the skin and on the rib cage
  clavicle  turned about the sternoclavicular centre and lengthened to meet the scapula at the acromioclavicular joint

The CT subject's shoulder girdle is narrower than the neutral MHR body (clavicle + scapula reach about 175 mm from the sternum, the
MHR shoulder joint is 191-210 mm away), so the girdle has to change size, not only turn. Joint centres (gh, ac) and the scapular rest
gaps in atlas.json are updated. Needs the MHR binding (tools/bind_mhr_joints.py) in atlas.json.

usage: python tools/seat_limbs_on_mhr.py atlas/atlas mhr_rest.npz [--dry-run]"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy.runtime import ellipsoid_gap  # noqa: E402
from reseat_bones import SkinDepth, hinge, outside_report, rot_about, sample  # noqa: E402

ELBOW_OVERSHOOT_M = 0.012        # the elbow joint centre lies a little short of the distal end of the humerus


def shortest_arc(a, b):
    """Rotation matrix taking direction a onto direction b by the smallest angle."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-12:
        return np.eye(3)
    return Rot.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas"); ap.add_argument("rest"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(a.atlas + ".json"))
    if not meta.get("mhr"):
        raise SystemExit("atlas.json has no MHR binding: run tools/bind_mhr_joints.py first")
    blob = bytearray(open(a.atlas + ".bin", "rb").read())
    rest = np.load(a.rest)
    depth = SkinDepth(np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3))
    by = {e["name"]: e for e in meta["structures"]}
    J = {j["name"]: j for j in meta["joints"]}
    mhr = meta["mhr"]
    rp = lambda j: np.asarray(mhr["rest_pos"][str(j)], np.float64)        # noqa: E731

    def get(name):
        e = by[name]
        return np.frombuffer(blob, np.float32, e["n_verts"] * 3, e["pos"]).reshape(-1, 3).astype(np.float64)

    def put(name, pos):
        e = by[name]
        blob[e["pos"]:e["pos"] + e["n_verts"] * 12] = pos.astype("<f4").tobytes()
        e["centroid"] = [round(float(v), 5) for v in pos.mean(0)]

    rows = []
    for k, side in (("l", "left"), ("r", "right")):
        sc, ac, gh = J[f"sc_{k}"], J[f"ac_{k}"], J[f"gh_{k}"]
        lb = mhr["limbs"][f"humerus_{side}"]
        shoulder, elbow = rp(lb["joint"]), rp(lb["child"])
        Jsc, Jac, Jgh = (np.asarray(j["center"], np.float64) for j in (sc, ac, gh))
        clav, scap, hum = get(f"clavicula_{side}"), get(f"scapula_{side}"), get(f"humerus_{side}")
        before = {n: outside_report(depth, p) for n, p in ((f"clavicula_{side}", clav), (f"scapula_{side}", scap), (f"humerus_{side}", hum))}

        # ---- humerus: a similarity that puts the head centre on the shoulder joint and the shaft on the shoulder->elbow line
        tip = hum[np.argmax(np.linalg.norm(hum - Jgh, axis=1))]
        scale_h = (np.linalg.norm(elbow - shoulder) + ELBOW_OVERSHOOT_M) / np.linalg.norm(tip - Jgh)
        R_h = shortest_arc(tip - Jgh, elbow - shoulder)
        # the shaft direction leaves the roll about the shaft free. The CT bone came with its own roll, so stay within 30 degrees
        # of it and take the roll that keeps the bone deepest under the arm skin (a free search flipped one humerus front to back)
        axis = (elbow - shoulder) / np.linalg.norm(elbow - shoulder)
        s_hum = sample(hum, 500)
        rolls = [Rot.from_rotvec(axis * np.radians(d)).as_matrix() @ R_h for d in range(-30, 31, 5)]
        cost = [float((hinge(depth(shoulder + scale_h * (s_hum - Jgh) @ R.T), 0.008) ** 2).sum()) for R in rolls]
        R_h = rolls[int(np.argmin(cost))]
        hum2 = shoulder + scale_h * (hum - Jgh) @ R_h.T

        # ---- scapula: glenoid on the shoulder joint; rotation about it and size are free. Clavicle: follows from the scapula.
        patch = np.asarray(ac["patch"]) if "patch" in ac else None
        ell_c, ell_r = (np.asarray(ac["ellipsoid"]["center"]), np.asarray(ac["ellipsoid"]["radii"])) if patch is not None else (None, None)
        s_scap, s_clav = sample(scap, 400), sample(clav, 250)

        def scap_tf(x, p):
            return shoulder + np.exp(x[3]) * (p - Jgh) @ Rot.from_rotvec(x[0:3]).as_matrix().T

        def clav_tf(x, p):
            ac2 = scap_tf(x, Jac[None])[0]
            s = np.linalg.norm(ac2 - Jsc) / np.linalg.norm(Jac - Jsc)
            return Jsc + s * (p - Jsc) @ shortest_arc(Jac - Jsc, ac2 - Jsc).T, ac2, s

        def residuals(x):
            sp = scap_tf(x, s_scap)
            cp, _, s_c = clav_tf(x, s_clav)
            r = [hinge(depth(sp), 0.005) * 1000 / np.sqrt(len(sp)) * 3.0,
                 hinge(depth(cp), 0.004) * 1000 / np.sqrt(len(cp)) * 3.0,
                 np.degrees(x[0:3]) * 0.10,
                 [x[3] * 100 * 0.05, (np.log(s_c) - x[3]) * 100 * 0.08]]          # modest size change, clavicle and scapula alike
            if patch is not None:                                                    # the scapula stays on the rib cage
                g, _ = ellipsoid_gap(scap_tf(x, scap[patch]), ell_c, ell_r)
                r.append([(g.mean() - ac["gap0"]) * 1000 * 0.5, max(ac["gapmin0"] - g.min(), 0.0) * 1000 * 0.5])
            return np.concatenate([np.atleast_1d(np.asarray(v, np.float64)).ravel() for v in r])

        best = None
        for seed in range(6):
            x0 = np.zeros(4) if seed == 0 else np.concatenate([np.random.default_rng(seed).normal(0, 0.2, 3), [0.1]])
            sol = least_squares(residuals, x0, method="trf", x_scale=0.1, max_nfev=400)
            if best is None or sol.cost < best.cost:
                best = sol
        x = best.x
        scap2 = scap_tf(x, scap)
        clav2, ac2, s_c = clav_tf(x, clav)
        for n, p in ((f"clavicula_{side}", clav2), (f"scapula_{side}", scap2), (f"humerus_{side}", hum2)):
            rows.append((n, before[n], outside_report(depth, p)))
            put(n, p)
        gh["center"] = [round(float(v), 6) for v in shoulder]
        ac["center"] = [round(float(v), 5) for v in ac2]
        if patch is not None:
            g, _ = ellipsoid_gap(scap2[patch], ell_c, ell_r)
            ac["gap0"], ac["gapmin0"] = round(float(g.mean()), 5), round(float(g.min()), 5)
        ax = hum2[np.argmax(np.linalg.norm(hum2 - shoulder, axis=1))] - shoulder
        print(f"{side:5s}: humerus x{scale_h:.3f}, turned {np.degrees(Rot.from_matrix(R_h).magnitude()):.0f} deg; now 0 mm from the shoulder joint, "
              f"shaft {np.degrees(np.arccos(ax @ (elbow - shoulder) / np.linalg.norm(ax) / np.linalg.norm(elbow - shoulder))):.1f} deg off the arm line | "
              f"scapula x{np.exp(x[3]):.3f}, turned {np.degrees(np.linalg.norm(x[0:3])):.0f} deg | clavicle x{s_c:.3f}")

    print(f"\n{'bone':18s} {'outside rest skin before':>26s} {'after':>22s}")
    for n, b, c in rows:
        print(f"{n:18s} {b[0]:8.1f}% worst {b[1]:5.1f} mm   {c[0]:8.1f}% worst {c[1]:5.1f} mm")
    if a.dry_run:
        print("\ndry run: nothing written")
        return
    meta.setdefault("history", []).append("tools/seat_limbs_on_mhr.py: humeri seated on the MHR shoulder-elbow joints, girdle re-sized to meet them")
    mhr["limb_seated"] = True
    json.dump(meta, open(a.atlas + ".json", "w"), separators=(",", ":"))
    open(a.atlas + ".bin", "wb").write(bytes(blob))
    print(f"\nwrote {a.atlas}.json and {a.atlas}.bin")


if __name__ == "__main__":
    main()
