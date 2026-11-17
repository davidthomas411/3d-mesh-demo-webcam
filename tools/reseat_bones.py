"""Containment rule at rest: put the shoulder girdle, the humeri and the skull of an atlas inside the MHR rest skin.

Why: the atlas build keeps soft tissue and the trunk bones under the skin, but the bones that hang off joints were never checked.
In the shipped atlas the humeri were 90% outside the rest skin (up to 10 cm), the clavicles 40-58%, the scapulae 15-20% and the skull 7%.
A bone that starts outside the skin stays outside in every pose, because the rig only copies skin motion onto it.

What it does (per side, joints stay pinned, bones stay rigid):
  clavicle  rotates about the sternoclavicular centre
  scapula   rides the clavicle (its acromioclavicular centre moves with it) and rotates about that centre;
            its contact patch is kept at the same gap from the rib-cage ellipsoid
  humerus   rides the scapula (glenohumeral centre moves with it) and rotates about that centre
  skull     rotates about the atlanto-occipital centre and may change size a little about it (the brain follows)
The rotations minimise how far sample points of each bone are from being at least `margin` under the rest skin, with a small
penalty on the rotation itself so that a bone already inside is left alone. Joint centres, centroids and the scapular rest gaps
in atlas.json are updated to match.

usage: python tools/reseat_bones.py atlas/atlas mhr_rest.npz [--out atlas/atlas] [--dry-run]
       mhr_rest.npz: verts (V,3) in the atlas rest frame (metres, y down, front -z), faces (F,3)"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anatomy.runtime import ellipsoid_gap  # noqa: E402

MARGIN_MM = {"humerus": 8.0, "clavicula": 4.0, "scapula": 5.0, "skull": 4.0}


class SkinDepth:
    """Signed depth under a closed skin mesh (metres, positive inside), from dense surface samples and their normals."""

    def __init__(self, verts, faces, n=600000, seed=0):
        tri = verts[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.maximum(area[:, None], 1e-12)
        rng = np.random.default_rng(seed)
        f = rng.choice(len(faces), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        self.p = tri[f, 0] + u[:, None] * (tri[f, 1] - tri[f, 0]) + v[:, None] * (tri[f, 2] - tri[f, 0])
        self.n = nrm[f]
        # outward normals: the centroid of a body mesh is inside it
        if ((self.p - verts.mean(0)) * self.n).sum(1).mean() < 0:
            self.n = -self.n
        self.tree = cKDTree(self.p)
        import trimesh
        self.mesh = trimesh.Trimesh(verts, faces, process=False)

    def __call__(self, x, exact=False):
        """exact=True decides inside/outside by ray casting (slow, right everywhere); the default uses the normal of the nearest
        surface sample, which is fast but wrong near thin inner surfaces such as the mouth cavity."""
        d, i = self.tree.query(x)
        inside = self.mesh.contains(x) if exact else ((x - self.p[i]) * self.n[i]).sum(1) < 0
        return np.where(inside, d, -d)


def rot_about(points, rotvec, centre, scale=1.0):
    return centre + scale * (points - centre) @ Rot.from_rotvec(rotvec).as_matrix().T


def hinge(depth, margin):
    return np.maximum(margin - depth, 0.0)


def outside_report(depth_fn, pos):
    d = depth_fn(pos, exact=True)
    return 100.0 * (d < 0).mean(), max(0.0, -d.min()) * 1000.0


def sample(pos, n, seed=0):
    if len(pos) <= n:
        return pos
    return pos[np.random.default_rng(seed).choice(len(pos), n, replace=False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas")
    ap.add_argument("rest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.load(open(a.atlas + ".json"))
    blob = bytearray(open(a.atlas + ".bin", "rb").read())
    rest = np.load(a.rest)
    depth = SkinDepth(np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3))
    by = {e["name"]: e for e in meta["structures"]}
    J = {j["name"]: j for j in meta["joints"]}

    def get(name):
        e = by[name]
        return np.frombuffer(blob, np.float32, e["n_verts"] * 3, e["pos"]).reshape(-1, 3).astype(np.float64)

    def put(name, pos):
        e = by[name]
        blob[e["pos"]:e["pos"] + e["n_verts"] * 12] = pos.astype("<f4").tobytes()
        e["centroid"] = [round(float(v), 5) for v in pos.mean(0)]

    before, after = {}, {}

    # ---- shoulder girdle and humerus, one side at a time ----
    for k, side in (("l", "left"), ("r", "right")):
        sc, ac, gh = J[f"sc_{k}"], J[f"ac_{k}"], J[f"gh_{k}"]
        n_clav, n_scap, n_hum = f"clavicula_{side}", f"scapula_{side}", f"humerus_{side}"
        clav, scap, hum = get(n_clav), get(n_scap), get(n_hum)
        Jsc, Jac, Jgh = (np.asarray(j["center"], np.float64) for j in (sc, ac, gh))
        patch = np.asarray(ac["patch"]) if "patch" in ac else None
        ell_c = np.asarray(ac["ellipsoid"]["center"]) if patch is not None else None
        ell_r = np.asarray(ac["ellipsoid"]["radii"]) if patch is not None else None
        for n, p in ((n_clav, clav), (n_scap, scap), (n_hum, hum)):
            before[n] = outside_report(depth, p)
        s_clav, s_scap, s_hum = sample(clav, 250), sample(scap, 400), sample(hum, 500)

        def place(x, clav_p, scap_p, hum_p, scap_patch=None):
            rc, rs, rh = x[0:3], x[3:6], x[6:9]
            c2 = rot_about(clav_p, rc, Jsc)
            ac2 = rot_about(Jac[None], rc, Jsc)[0]
            s_rigid = lambda p: rot_about(rot_about(p, rc, Jsc), rs, ac2)       # noqa: E731  ride the clavicle, then turn about AC
            s2 = s_rigid(scap_p)
            gh2 = s_rigid(Jgh[None])[0]
            h2 = rot_about(s_rigid(hum_p), rh, gh2)
            pp = s_rigid(scap_patch) if scap_patch is not None else None
            return c2, s2, h2, ac2, gh2, pp

        def residuals(x):
            c2, s2, h2, _, _, pp = place(x, s_clav, s_scap, s_hum, scap[patch] if patch is not None else None)
            r = [hinge(depth(c2), MARGIN_MM["clavicula"] / 1000) * 1000 / np.sqrt(len(c2)) * 3.0,
                 hinge(depth(s2), MARGIN_MM["scapula"] / 1000) * 1000 / np.sqrt(len(s2)) * 3.0,
                 hinge(depth(h2), MARGIN_MM["humerus"] / 1000) * 1000 / np.sqrt(len(h2)) * 3.0,
                 np.degrees(x[0:3]) * 0.04, np.degrees(x[3:6]) * 0.04, np.degrees(x[6:9]) * 0.01]
            if pp is not None:                                                     # stay on the rib cage
                g, _ = ellipsoid_gap(pp, ell_c, ell_r)
                r.append(np.array([(g.mean() - ac["gap0"]) * 1000 * 0.5, max(ac["gapmin0"] - g.min(), 0.0) * 1000 * 0.5]))
            return np.concatenate(r)

        best = None
        for seed in range(6):                                                      # a few starts: the hinge cost has flat regions
            x0 = np.zeros(9) if seed == 0 else np.random.default_rng(seed).normal(0, 0.25, 9)
            sol = least_squares(residuals, x0, method="trf", x_scale=0.1, max_nfev=400)
            if best is None or sol.cost < best.cost:
                best = sol
        x = best.x
        c2, s2, h2, ac2, gh2, pp = place(x, clav, scap, hum, scap[patch] if patch is not None else None)
        for n, p in ((n_clav, c2), (n_scap, s2), (n_hum, h2)):
            after[n] = outside_report(depth, p)
            put(n, p)
        ac["center"] = [round(float(v), 5) for v in ac2]
        gh["center"] = [round(float(v), 5) for v in gh2]
        if pp is not None:
            g, _ = ellipsoid_gap(pp, ell_c, ell_r)
            ac["gap0"], ac["gapmin0"] = round(float(g.mean()), 5), round(float(g.min()), 5)
        print(f"{side:5s}: clavicle turned {np.degrees(np.linalg.norm(x[0:3])):5.1f} deg, scapula {np.degrees(np.linalg.norm(x[3:6])):5.1f} deg, "
              f"humerus {np.degrees(np.linalg.norm(x[6:9])):5.1f} deg; glenohumeral centre moved {np.linalg.norm(gh2 - Jgh) * 1000:.0f} mm, "
              f"now {depth(gh2[None])[0] * 1000:.0f} mm under the skin")

    # ---- skull (and whatever is rigid with it) ----
    if "ao" in J:
        Jao = np.asarray(J["ao"]["center"], np.float64)
        skull = get("skull")
        before["skull"] = outside_report(depth, skull)
        s_skull = sample(skull, 600)

        def res_skull(x):
            p = rot_about(s_skull, x[0:3], Jao, np.exp(x[3]))
            return np.concatenate([hinge(depth(p, exact=True), MARGIN_MM["skull"] / 1000) * 1000 / np.sqrt(len(p)) * 3.0, np.degrees(x[0:3]) * 0.05, [x[3] * 100 * 0.15]])

        sol = least_squares(res_skull, np.zeros(4), method="trf", x_scale=0.05, diff_step=2e-3, max_nfev=60)
        x = sol.x
        for e in meta["structures"]:
            if e["name"] == "skull" or (e.get("bind") == "frame" and e.get("frame") == "skull"):
                put(e["name"], rot_about(get(e["name"]), x[0:3], Jao, np.exp(x[3])))
        after["skull"] = outside_report(depth, get("skull"))
        print(f"skull: turned {np.degrees(np.linalg.norm(x[0:3])):.1f} deg about the atlanto-occipital centre, size x{np.exp(x[3]):.3f}")

    print(f"\n{'bone':18s} {'outside before':>22s} {'outside after':>22s}")
    for n in before:
        print(f"{n:18s} {before[n][0]:6.1f}% worst {before[n][1]:5.1f} mm   {after[n][0]:6.1f}% worst {after[n][1]:5.1f} mm")

    if a.dry_run:
        print("\ndry run: nothing written")
        return
    out = a.out or a.atlas
    meta.setdefault("history", []).append("tools/reseat_bones.py: shoulder girdle, humeri and skull re-seated inside the rest skin")
    json.dump(meta, open(out + ".json", "w"), separators=(",", ":"))
    open(out + ".bin", "wb").write(bytes(blob))
    print(f"\nwrote {out}.json and {out}.bin")


if __name__ == "__main__":
    main()
