"""Bind the rig's segments to joints of the MHR skeleton, so the rig can be driven by the joints the estimator already predicts
instead of re-estimating each segment from a patch of skin.

For each rig segment (pelvis, lumbar, thorax, head, upper arms, thighs, shoulders) the driving joint is the MHR joint that carries
most of the linear-blend-skinning weight of that segment's skin patch, so the choice follows the body model's own rigging and needs
no joint names. Where that joint is a leaf (a twist helper along a limb) its parent, the limb's own joint, is used. For the limb bones the child joint down the limb gives a bone length, which scales the bone to the person.

Writes atlas.json["mhr"]:
  segments  {segment name: joint index}
  rest_pos  {joint index: [x, y, z]}   rest joint positions, atlas frame (metres, y down, front -z)
  rest_rot  {joint index: [9 floats]}  rest global joint rotations, row-major, atlas frame
  limbs     {bone name: {"joint": j, "child": c, "rest_len": metres}}

usage: python tools/bind_mhr_joints.py atlas/atlas path/to/mhr_model.pt [--clips clips/*.npz] [--dry-run]
       --clips also reports how far the joint-driven segment frames are from the skin-patch fits on real motion."""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anatomy.runtime import Atlas, kabsch  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])            # MHR model frame -> atlas / estimator output frame
LIMB_BONES = {"humerus_left": "uparm_l", "humerus_right": "uparm_r", "femur_left": "thigh_l", "femur_right": "thigh_r"}


def quat_to_rot(q):                           # (..., 4) as x, y, z, w
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


class MHR:
    def __init__(self, path):
        import torch
        self.torch = torch
        self.m = torch.jit.load(path, map_location="cpu").eval()
        sk = self.m.character_torch.skeleton
        self.parents = sk.joint_parents.numpy().astype(int)
        lbs = self.m.character_torch.linear_blend_skinning
        self.w_joint = lbs.skin_indices_flattened.numpy().astype(int)
        self.w_vert = lbs.vert_indices_flattened.numpy().astype(int)
        self.w = lbs.skin_weights_flattened.numpy().astype(np.float64)

    def pose(self, params=None, shape=None):
        """-> verts (V,3), joint positions (J,3), joint rotations (J,3,3), all in the atlas frame (metres)."""
        t = self.torch
        p = t.zeros(1, 204) if params is None else t.as_tensor(np.asarray(params, np.float32)).reshape(1, -1)
        s = t.zeros(1, 45) if shape is None else t.as_tensor(np.asarray(shape, np.float32)).reshape(1, -1)
        with t.no_grad():
            v, sk = self.m(s, p, t.zeros(1, 72), False)
        v = v[0].numpy().astype(np.float64) * 0.01 @ FLIP
        sk = sk[0].numpy().astype(np.float64)
        return v, sk[:, :3] * 0.01 @ FLIP, FLIP @ quat_to_rot(sk[:, 3:7]) @ FLIP

    def children(self, j):
        return [c for c in range(len(self.parents)) if self.parents[c] == j]


def joint_frame(rest_pos, rest_rot, pos, rot, j):
    """Rigid rest->live transform of joint j as (R, c_rest, c_live): x -> R (x - c_rest) + c_live."""
    return rot[j] @ rest_rot[j].T, rest_pos[j], pos[j]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas")
    ap.add_argument("mhr")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    atlas = Atlas(a.atlas)
    mhr = MHR(a.mhr)
    v0, p0, r0 = mhr.pose()
    err = max(np.abs(v0[sg["idx"]] - sg["rest"]).max() for sg in atlas.segments)
    print(f"neutral MHR mesh vs atlas rest patches: {err * 1000:.2f} mm (must be ~0)")
    assert err < 1e-3, "this MHR model is not the one the atlas was built on"

    # dominant joint of each segment's skin patch
    nJ = len(mhr.parents)
    seg_joint = {}
    for sg in atlas.segments:
        sel = np.isin(mhr.w_vert, sg["idx"])
        tot = np.bincount(mhr.w_joint[sel], weights=mhr.w[sel], minlength=nJ)
        order = np.argsort(-tot)[:3]
        j = int(order[0])
        if not mhr.children(j):
            # a leaf here is a twist helper spread along a limb (it carries the skin's partial twist). The bone turns with
            # the limb's own joint, its parent.
            j = int(mhr.parents[j])
        seg_joint[sg["name"]] = j
        print(f"  {sg['name']:11s} -> joint {j:3d} (skin patch dominated by joint {order[0]:3d}: {100 * tot[order[0]] / tot.sum():4.1f}% of the patch's skinning weight; next {order[1]} {100 * tot[order[1]] / tot.sum():.1f}%, {order[2]} {100 * tot[order[2]] / tot.sum():.1f}%)")

    # limb bones: length from the driving joint to the child that continues down the limb (the child farthest away)
    limbs = {}
    for bone, seg in LIMB_BONES.items():
        j = seg_joint[seg]
        kids = mhr.children(j)
        if not kids:
            continue
        kids = [k for k in kids if mhr.children(k)] or kids       # the next real joint down the limb, not a twist helper
        c = max(kids, key=lambda k: np.linalg.norm(p0[k] - p0[j]))
        limbs[bone] = {"joint": j, "child": int(c), "rest_len": round(float(np.linalg.norm(p0[c] - p0[j])), 5)}
        print(f"  {bone:14s} length joint {j} -> {c}: {limbs[bone]['rest_len'] * 1000:.0f} mm at rest")

    used = sorted(set(seg_joint.values()) | {l["child"] for l in limbs.values()})
    block = {"n_joints": nJ, "segments": seg_joint,
             "rest_pos": {str(j): [round(float(x), 6) for x in p0[j]] for j in used},
             "rest_rot": {str(j): [round(float(x), 7) for x in r0[j].reshape(-1)] for j in used},
             "limbs": limbs}

    # how close is the joint-driven frame to the skin-patch fit on real motion?
    if a.clips:
        from mhr_clip import load_clip, prepare
        from scipy.spatial.transform import Rotation as Rot
        stats = {sg["name"]: [] for sg in atlas.segments}
        for pattern in a.clips:
            for f in glob.glob(pattern):
                arr, fps, _ = load_clip(f)
                P = prepare(arr)
                for k in np.linspace(0, len(P) - 1, min(25, len(P))).astype(int):
                    v, p, r = mhr.pose(P[k])
                    for sg in atlas.segments:
                        Rk, cs, cd = kabsch(sg["rest"], v[sg["idx"]])
                        Rj, c_rest, c_live = joint_frame(p0, r0, p, r, seg_joint[sg["name"]])
                        ang = np.degrees(Rot.from_matrix(Rk.T @ Rj).magnitude())
                        pos = np.linalg.norm((Rj @ (cs - c_rest) + c_live) - cd) * 1000      # where each puts the patch centroid
                        stats[sg["name"]].append((ang, pos))
        print("\njoint-driven frame vs skin-patch fit on the clips (rotation difference, patch-centroid difference):")
        for n, s in stats.items():
            s = np.array(s)
            print(f"  {n:11s} rotation median {np.median(s[:, 0]):5.1f} deg  p95 {np.percentile(s[:, 0], 95):5.1f} deg | position median {np.median(s[:, 1]):5.1f} mm  p95 {np.percentile(s[:, 1], 95):5.1f} mm")

    if a.dry_run:
        print("\ndry run: nothing written")
        return
    meta = json.load(open(a.atlas + ".json"))
    meta["mhr"] = block
    json.dump(meta, open(a.atlas + ".json", "w"), separators=(",", ":"))
    print(f"\nwrote the mhr binding into {a.atlas}.json ({len(used)} joints)")


if __name__ == "__main__":
    main()
