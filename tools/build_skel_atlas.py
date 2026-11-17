"""Build a skeleton atlas from SKEL fitted to the MHR body (tools/fit_skel_to_mhr.py), in the same format as the CT atlas, so the
live rig can draw SKEL's learned skeleton: skull, separate jaw, neck, rib cage, lumbar spine, pelvis, scapulae, and the limb bones the
CT atlas does not have (forearms, hands, femora, lower legs, feet).

How each bone is driven:
  trunk and head   rigid with the rig's skin-anchored segments (blended between neighbours along the neck and the lumbar spine)
  limb bones       rigid with a joint of the MHR skeleton ("mhr" bind): the bone keeps the place it has relative to that joint in the
                   rest fit, turns with the joint and is scaled by the joint-to-joint length
  jaw              rigid with the head, plus a hinge about the line through its two condyles (driven by a face tracker in the page)

SKEL is under a non-commercial research licence. The output (atlas/skel.*) is derived from it and is git-ignored.

usage: python tools/build_skel_atlas.py skel_fit.npz atlas/atlas mhr_model.pt mhr_rest.npz atlas/skel"""
import argparse, json, os, sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from anatomy.runtime import Atlas  # noqa: E402
from bind_mhr_joints import MHR  # noqa: E402
from reseat_bones import SkinDepth, hinge  # noqa: E402
from seat_limbs_on_mhr import shortest_arc  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

BONE_COLOR = [0.93, 0.9, 0.82]
TARGET_TRIS = {"thorax": 9000, "skull": 5000, "jaw": 1500, "hand": 1800, "pelvis": 4000, "default": 1600}


def welded(verts, faces):
    """Vertices merged by position (10 micrometres) and repeated triangles dropped: SKEL stores some bones as loose triangles,
    and the right hand twice."""
    uniq, inv = np.unique(np.round(verts, 5), axis=0, return_inverse=True)
    f = inv.reshape(-1)[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    m = trimesh.Trimesh(uniq, f[np.sort(first)], process=False)
    m.remove_unreferenced_vertices()
    return m


def decimate(m, target):
    if len(m.faces) <= target:
        return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    import fast_simplification
    v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32), target_count=int(target))
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def main_child(mhr, p0, j):
    kids = [c for c in mhr.children(j) if mhr.children(c)]
    return max(kids, key=lambda c: np.linalg.norm(p0[c] - p0[j])) if kids else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("skel_fit"); ap.add_argument("base_atlas"); ap.add_argument("mhr"); ap.add_argument("rest"); ap.add_argument("out")
    a = ap.parse_args()
    z = np.load(a.skel_fit, allow_pickle=True)
    V, Fc = z["skel_verts"].astype(np.float64), z["skel_faces"]
    names = [str(n) for n in z["bone_names"]]
    bvi = z["bone_verts_indices"]
    base = Atlas(a.base_atlas)
    seg = {g["name"]: i for i, g in enumerate(base.segments)}
    mhr = MHR(a.mhr)
    v_rest, p0, r0 = mhr.pose()
    rest = np.load(a.rest)
    depth = SkinDepth(np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3))
    bm = base.meta["mhr"]
    lev = base.meta["spine_levels"]

    def bone_mesh(n):
        s, e = bvi[names.index(n)]
        f = Fc[(Fc[:, 0] >= s) & (Fc[:, 0] < e)] - s
        return welded(V[s:e], f)

    # MHR joint chains down the limbs: shoulder -> elbow -> wrist, hip -> knee -> ankle -> toes
    chain = {}
    for side, k in (("l", "left"), ("r", "right")):
        sh = bm["limbs"][f"humerus_{k}"]["joint"]; el = main_child(mhr, p0, sh); wr = main_child(mhr, p0, el)
        hp = bm["limbs"][f"femur_{k}"]["joint"]; kn = main_child(mhr, p0, hp); an = main_child(mhr, p0, kn)
        toe = an
        for _ in range(2):                                   # the next joint that is actually somewhere else (the toes)
            nxt = main_child(mhr, p0, toe)
            if nxt is None:
                break
            toe = nxt
            if np.linalg.norm(p0[toe] - p0[an]) > 0.02:
                break
        chain[side] = dict(shoulder=sh, elbow=el, wrist=wr, hip=hp, knee=kn, ankle=an, toe=toe)
    used_joints = set()
    structures = []

    def add(name, verts, faces, group, bind, **extra):
        structures.append(dict(name=name, verts=verts, faces=faces, group=group, bind=bind, **extra))

    dominant = np.full(len(v_rest), -1); best_w = np.zeros(len(v_rest))
    for jj, vv, ww in zip(mhr.w_joint, mhr.w_vert, mhr.w):
        if ww > best_w[vv]:
            best_w[vv] = ww; dominant[vv] = jj

    def mhr_bind(j, child=None, girth=False):
        """Bone rigid with MHR joint j. With a child joint its LENGTH follows the joint-to-child distance. With girth=True its
        THICKNESS follows the limb: a sample of the skin carried by j (and its twist helpers) whose RMS distance from the
        joint-to-child line is compared with the rest value at run time."""
        used_joints.update([j] + ([child] if child is not None else []))
        d = {"joint": int(j)}
        if child is not None:
            d.update(child=int(child), rest_len=round(float(np.linalg.norm(p0[child] - p0[j])), 5))
        if girth and child is not None:
            owners = [j] + [k for k in mhr.children(j) if not mhr.children(k)]
            idx = np.where(np.isin(dominant, owners))[0]
            ax = (p0[child] - p0[j]) / np.linalg.norm(p0[child] - p0[j])
            t = (v_rest[idx] - p0[j]) @ ax
            idx = idx[(t > 0.15 * d["rest_len"]) & (t < 0.85 * d["rest_len"])]          # the shaft, away from the joints
            if len(idx) >= 12:
                idx = idx[np.linspace(0, len(idx) - 1, min(120, len(idx))).astype(int)]
                rel = v_rest[idx] - p0[j]; rad = rel - np.outer(rel @ ax, ax)
                d.update(girth_idx=[int(i) for i in idx], girth_rest=round(float(np.sqrt((rad ** 2).sum(1).mean())), 5),
                         girth_pos=[[round(float(x), 5) for x in v_rest[i]] for i in idx])        # rest positions (tests, checks)
        return d

    def along(y, lo, hi):                                    # 0 at lo, 1 at hi (y is down, values are negative)
        return float(np.clip((y - lo) / (hi - lo), 0.0, 1.0))

    # ---- head: skull, jaw, neck vertebrae
    head = bone_mesh("head")
    comps = sorted(head.split(only_watertight=False), key=lambda m: -len(m.vertices))
    front = [m for m in comps if m.vertices[:, 2].mean() < -0.03]
    skull = max(front, key=lambda m: np.ptp(m.vertices[:, 1]) * np.ptp(m.vertices[:, 2]))
    low = [m for m in front if m is not skull and m.vertices[:, 1].mean() > skull.vertices[:, 1].mean()]
    jaw = max(low, key=lambda m: len(m.vertices))
    jaw_parts, skull_parts, neck = [jaw], [skull], []
    for m in comps:
        if m is skull or m is jaw:
            continue
        c = m.vertices.mean(0)
        if c[2] < -0.03:                                     # teeth and small pieces: with whichever of the two they are nearer
            dj = np.linalg.norm(jaw.vertices - c, axis=1).min(); ds = np.linalg.norm(skull.vertices - c, axis=1).min()
            (jaw_parts if dj < ds else skull_parts).append(m)
        else:
            neck.append(m)
    skull_m = trimesh.util.concatenate(skull_parts); jaw_m = trimesh.util.concatenate(jaw_parts)
    v, f = decimate(skull_m, TARGET_TRIS["skull"]); add("skel_skull", v, f, "skel_axial", "bone", bone_w=[[seg["head"], 1.0]], shell_mm=6.5)   # SKEL's skull is one closed surface with no cranial cavity: a radiograph must treat it as a shell
    jv, jf = decimate(jaw_m, TARGET_TRIS["jaw"])
    # hinge through the two condyles: on each side, the highest point of the back of the jaw
    cond = []
    for sgn in (1, -1):
        s = jv[(jv[:, 0] * sgn > 0.02) & (jv[:, 2] > np.percentile(jv[:, 2], 70))]
        cond.append(s[np.argsort(s[:, 1])[:max(3, len(s) // 20)]].mean(0))
    axis = (cond[0] - cond[1]) / np.linalg.norm(cond[0] - cond[1])
    add("skel_jaw", jv, jf, "skel_axial", "bone", bone_w=[[seg["head"], 1.0]],
        jaw={"pivot": [round(float(x), 5) for x in (cond[0] + cond[1]) / 2], "axis": [round(float(x), 5) for x in axis]})
    neck = sorted([m for m in neck if len(m.vertices) > 30], key=lambda m: m.vertices[:, 1].mean())
    y_top, y_bot = (neck[0].vertices[:, 1].mean(), neck[-1].vertices[:, 1].mean()) if neck else (0, 1)
    for i, m in enumerate(neck):                             # top (C1) rides the head, bottom rides the upper thorax
        w_head = 1.0 - along(m.vertices[:, 1].mean(), y_top, y_bot) * 0.85
        v, f = decimate(m, 500)
        add(f"skel_cervical_{i + 1}", v, f, "skel_axial", "bone", bone_w=[[seg["head"], round(w_head, 3)], [seg["thorax_up"], round(1 - w_head, 3)]])

    # ---- trunk
    v, f = decimate(bone_mesh("thorax"), TARGET_TRIS["thorax"])
    add("skel_thorax", v, f, "skel_axial", "bone", bone_w=[[seg["thorax_low"], 0.5], [seg["thorax_up"], 0.5]])
    for i, m in enumerate(sorted(bone_mesh("lumbar_body").split(only_watertight=False), key=lambda m: m.vertices[:, 1].mean())):
        y = m.vertices[:, 1].mean()                          # from the lower thorax (top) through the lumbar segment to the pelvis
        if y < lev["lumbar"]:
            t = along(y, lev["thorax_low"], lev["lumbar"]); w = [[seg["thorax_low"], round(1 - t, 3)], [seg["lumbar"], round(t, 3)]]
        else:
            t = along(y, lev["lumbar"], lev["pelvis"]); w = [[seg["lumbar"], round(1 - t, 3)], [seg["pelvis"], round(t, 3)]]
        v, f = decimate(m, 700)
        add(f"skel_lumbar_{i + 1}", v, f, "skel_axial", "bone", bone_w=[x for x in w if x[1] > 0])
    v, f = decimate(bone_mesh("pelvis"), TARGET_TRIS["pelvis"]); add("skel_pelvis", v, f, "skel_axial", "bone", bone_w=[[seg["pelvis"], 1.0]])

    # ---- limbs: every bone is DEFINED ON THE BODY MODEL'S SKELETON. SKEL's own fit leaves the extremities outside the MHR skin
    # (feet 35-53%, toes 40%, hands 17-25%, ulnae 16-19%), so each bone's SKEL joint segment (hip->knee, a phalanx's two ends, ...)
    # is mapped onto the matching MHR joint segment by a similarity, and the bone is then rigid with that MHR joint.
    J = {n: z["joints"][i].astype(np.float64) for i, n in enumerate(names)}
    wsum = np.bincount(mhr.w_joint, weights=mhr.w, minlength=len(mhr.parents))

    def seg_map(a0, a1, b0, b1, lo=0.75, hi=1.35):
        """Similarity taking segment a0->a1 onto b0->b1: x -> scale R (x - a0) + b0."""
        k = float(np.clip(np.linalg.norm(b1 - b0) / max(np.linalg.norm(a1 - a0), 1e-9), lo, hi))
        return k, shortest_arc(a1 - a0, b1 - b0), a0, b0

    def apply(tf, p):
        k, R, a0, b0 = tf
        return b0 + k * (p - a0) @ R.T

    def pieces(n):
        return sorted(bone_mesh(n).split(only_watertight=False), key=lambda m: -len(m.vertices))

    def pca_ends(p):
        c = p.mean(0); _, sv, vt = np.linalg.svd(p - c, full_matrices=False)
        t = (p - c) @ vt[0]
        return p[np.argmin(t)], p[np.argmax(t)], float(sv[0] / max(sv[1], 1e-9)), float(np.ptp(t))

    def chain_down(j):
        out = [j]
        while len(mhr.children(out[-1])) == 1:
            out.append(mhr.children(out[-1])[0])
        return out

    report = []
    for side in ("l", "r"):
        c = chain[side]
        v, f = decimate(bone_mesh(f"scapula_{side}"), TARGET_TRIS["default"])
        add(f"skel_scapula_{side}", v, f, "skel_arm", "bone", bone_w=[[seg[f"shoulder_{side}"], 1.0]])

        # long bones: SKEL joint segment -> MHR joint segment
        for bones, group, ja, jb, j, child in (((f"humerus_{side}",), "skel_arm", f"humerus_{side}", f"ulna_{side}", c["shoulder"], c["elbow"]),
                                              ((f"ulna_{side}", f"radius_{side}"), "skel_limbs", f"ulna_{side}", f"hand_{side}", c["elbow"], c["wrist"]),
                                              ((f"femur_{side}",), "skel_limbs", f"femur_{side}", f"tibia_{side}", c["hip"], c["knee"]),
                                              ((f"tibia_{side}",), "skel_limbs", f"tibia_{side}", f"talus_{side}", c["knee"], c["ankle"])):
            tf = seg_map(J[ja], J[jb], p0[j], p0[child])
            for bone in bones:
                m = bone_mesh(bone); m.vertices = apply(tf, m.vertices)
                v, f = decimate(m, TARGET_TRIS["default"])
                add("skel_" + bone, v, f, group, "mhr", mhr=mhr_bind(j, child, girth=True))

        # ---- hand: wrist block rigid with the wrist; every metacarpal and phalanx on its own MHR finger segment
        wrist = mhr.children(c["wrist"])[0] if len(mhr.children(c["wrist"])) == 1 else c["wrist"]
        fingers = [[wrist] + chain_down(r) for r in mhr.children(wrist)]
        mean_dir = np.mean([p0[f[-1]] - p0[wrist] for f in fingers], 0)
        dev = [np.degrees(np.arccos(np.clip((p0[f[-1]] - p0[wrist]) @ mean_dir / np.linalg.norm(p0[f[-1]] - p0[wrist]) / np.linalg.norm(mean_dir), -1, 1))) for f in fingers]
        thumb = int(np.argmax(dev))
        knuckle = lambda f, n: p0[f[-(n + 1)]]                # start of the last n segments  # noqa: E731
        others = sorted([i for i in range(len(fingers)) if i != thumb], key=lambda i: np.linalg.norm(knuckle(fingers[i], 3) - knuckle(fingers[thumb], 2)))
        hp = pieces(f"hand_{side}")
        swrist = J[f"hand_{side}"]
        info = [pca_ends(m.vertices) for m in hp]
        long_ids = [i for i, (e0, e1, el, ln) in enumerate(info) if el > 1.6 and ln > 0.011]
        prox = {i: (info[i][0], info[i][1]) if np.linalg.norm(info[i][0] - swrist) < np.linalg.norm(info[i][1] - swrist) else (info[i][1], info[i][0]) for i in long_ids}
        starts = sorted(long_ids, key=lambda i: np.linalg.norm(prox[i][0] - swrist))[:len(fingers)]
        used = set(starts); schains = []
        for st in starts:                                       # follow each metacarpal out to the fingertip
            ch = [st]
            while True:
                tip = prox[ch[-1]][1]
                cand = [i for i in long_ids if i not in used and np.linalg.norm(prox[i][0] - tip) < 0.014]
                if not cand:
                    break
                nxt = min(cand, key=lambda i: np.linalg.norm(prox[i][0] - tip)); used.add(nxt); ch.append(nxt)
            schains.append(ch)
        sthumb = min(range(len(schains)), key=lambda i: len(schains[i]))
        sothers = sorted([i for i in range(len(schains)) if i != sthumb], key=lambda i: np.linalg.norm(prox[schains[i][0]][1] - prox[schains[sthumb][0]][1]))
        seated = set()
        for si, mi in [(sthumb, thumb)] + list(zip(sothers, others)):
            ch, fj = schains[si], fingers[mi]
            segs = [(x, y) for x, y in zip(fj[:-1], fj[1:]) if np.linalg.norm(p0[y] - p0[x]) > 1e-4][-len(ch):]
            for piece, (ja, jb) in zip(ch, segs):
                tf = seg_map(prox[piece][0], prox[piece][1], p0[ja], p0[jb], 0.6, 1.6)
                add(f"skel_hand_{side}_{mi}_{ja}", apply(tf, hp[piece].vertices), np.asarray(hp[piece].faces), "skel_limbs", "mhr", mhr=mhr_bind(ja, jb))
                seated.add(piece)
        rest_ids = [i for i in range(len(hp)) if i not in seated]
        if rest_ids:                                            # carpals: one block, carried by the wrist, turned to face the knuckles
            blk = trimesh.util.concatenate([hp[i] for i in rest_ids])
            s_kn = np.mean([prox[ch[0]][1] for ch in schains], 0)
            m_kn = np.mean([knuckle(fingers[i], 3 if i != thumb else 2) for i in range(len(fingers))], 0)
            tf = seg_map(swrist, s_kn, p0[wrist], m_kn, 0.8, 1.25)
            add(f"skel_carpals_{side}", apply(tf, blk.vertices), np.asarray(blk.faces), "skel_limbs", "mhr", mhr=mhr_bind(wrist))
        report.append(f"hand_{side}: {len(seated)} finger bones on MHR finger joints (chain lengths {[len(x) for x in schains]}), {len(rest_ids)} wrist bones as one block")

        # ---- foot: SKEL's ankle -> toe-joint line onto the MHR one, rolled about that line to sit inside the foot; then each
        # foot bone goes with the MHR joint that carries the skin next to it
        foot_j = chain_down(c["ankle"])
        toe_j = max(foot_j, key=lambda k: np.linalg.norm(p0[k] - p0[c["ankle"]]) if wsum[k] > 0 else -1)
        fparts = [(n, m) for n in (f"talus_{side}", f"calcn_{side}", f"toes_{side}") for m in pieces(n)]
        foot_base = seg_map(J[f"talus_{side}"], J[f"toes_{side}"], p0[c["ankle"]], p0[toe_j], 0.8, 1.25)
        allv = np.vstack([m.vertices for _, m in fparts]); samp = apply(foot_base, allv[::max(1, len(allv) // 900)])
        axis = (p0[toe_j] - p0[c["ankle"]]) / np.linalg.norm(p0[toe_j] - p0[c["ankle"]])
        # then a small rigid adjustment (turn, shift, size) about the ankle so the whole foot skeleton sits under the skin
        from scipy.optimize import least_squares
        o = p0[c["ankle"]]

        def adjust(x, q):
            return o + x[3:6] + np.exp(x[6]) * (q - o) @ Rot.from_rotvec(x[0:3]).as_matrix().T

        def res(x):
            return np.concatenate([hinge(depth(adjust(x, samp)), 0.004) * 1000 / np.sqrt(len(samp)) * 4.0,
                                   np.degrees(x[0:3]) * 0.02, x[3:6] * 1000 * 0.03, [x[6] * 100 * 0.05]])

        best = None
        for start_deg in (0, -25, 25):                         # the roll about the foot's own axis is the least certain part
            x0 = np.r_[axis * np.radians(start_deg), 0, 0, 0, 0]
            sol = least_squares(res, x0, method="trf", x_scale=0.05, max_nfev=300,
                                bounds=(np.r_[-1.0, -1.0, -1.0, -0.03, -0.03, -0.03, -0.2], np.r_[1.0, 1.0, 1.0, 0.03, 0.03, 0.03, 0.15]))
            if best is None or sol.cost < best.cost:
                best = sol
        xf = best.x

        def foot_tf(p, tf=foot_base, xf=xf):
            return adjust(xf, apply(tf, p))

        foot_skin = np.where(np.isin(mhr.w_joint, foot_j))[0]
        by_joint = {}
        for n, m in fparts:
            pv = foot_tf(m.vertices); cen = pv.mean(0)
            near = foot_skin[np.argsort(np.linalg.norm(v_rest[mhr.w_vert[foot_skin]] - cen, axis=1))[:60]]
            tot = np.bincount(mhr.w_joint[near], weights=mhr.w[near], minlength=len(mhr.parents))
            by_joint.setdefault(int(np.argmax(tot)), []).append(trimesh.Trimesh(pv, m.faces, process=False))
        for jf, ms in by_joint.items():
            v, f = decimate(trimesh.util.concatenate(ms), 2200)
            add(f"skel_foot_{side}_{jf}", v, f, "skel_limbs", "mhr", mhr=mhr_bind(jf))
        report.append(f"foot_{side}: turned {np.degrees(np.linalg.norm(xf[0:3])):.0f} deg, moved {np.linalg.norm(xf[3:6]) * 1000:.0f} mm, size x{np.exp(xf[6]):.2f}; bones per MHR joint {dict((k, len(v)) for k, v in by_joint.items())}")

    # ---- write: same layout as the CT atlas (anatomy.atlas_build)
    blob = bytearray()

    def put(arr):
        if len(blob) % 4:
            blob.extend(b"\0" * (4 - len(blob) % 4))
        off = len(blob); blob.extend(np.ascontiguousarray(arr).tobytes()); return off

    segs = []
    for g in base.segments:
        segs.append({"name": g["name"], "n": int(len(g["idx"])), "idx": put(g["idx"].astype("<u2")), "rest": put(g["rest"].astype("<f4"))})
    out_structs = []
    for s in structures:
        v, f = s["verts"], s["faces"]
        u16 = len(v) < 65536
        e = {"name": s["name"], "display": s["name"].replace("skel_", "").replace("_", " "), "group": s["group"], "color": BONE_COLOR, "uncertainty_mm": 15,
             "n_verts": int(len(v)), "n_tris": int(len(f)), "volume_ml": 0.0, "centroid": [round(float(x), 5) for x in v.mean(0)],
             "pos": put(v.astype("<f4")), "idx_type": "u16" if u16 else "u32", "idx": put(f.astype("<u2" if u16 else "<u4")), "bind": s["bind"]}
        for k in ("bone_w", "mhr", "jaw", "shell_mm"):
            if k in s:
                e[k] = s[k]
        out_structs.append(e)
    mb = json.loads(json.dumps(bm))
    for j in sorted(used_joints):
        mb["rest_pos"][str(j)] = [round(float(x), 6) for x in p0[j]]
        mb["rest_rot"][str(j)] = [round(float(x), 7) for x in r0[j].reshape(-1)]
    meta = {"version": 2, "frame": base.meta["frame"], "n_skin_vertices": base.meta["n_skin_vertices"], "segments": segs, "spine_levels": lev,
            "chain": [], "joints": [], "disclaimer": "SKEL skeleton fitted to the MHR body. Generic, not patient-specific. Not for clinical use.",
            "rig": "SKEL bones: trunk rigid with skin-anchored segments, limbs rigid with MHR joints, jaw hinged",
            "structures": out_structs, "source": {"dataset": "SKEL v1.1 (Keller et al. 2023), non-commercial research licence", "subject": str(z["gender"])},
            "landmarks_mm": None, "mhr": mb}
    json.dump(meta, open(a.out + ".json", "w"), separators=(",", ":"))
    open(a.out + ".bin", "wb").write(bytes(blob))
    tris = sum(e["n_tris"] for e in out_structs)
    print(f"wrote {a.out}.json/.bin: {len(out_structs)} bones, {tris} triangles, {len(blob) / 1e6:.1f} MB")
    jw = next(e for e in out_structs if e["name"] == "skel_jaw")["jaw"]
    print("  head split into skull %d, jaw %d and %d neck vertebrae; jaw hinge axis %s through %s" % (len(skull_m.vertices), len(jaw_m.vertices), len(neck), jw["axis"], jw["pivot"]))
    print("  limb joints:", chain)
    for line in report:
        print("  " + line)
    agg = {}
    for st in structures:
        if st["bind"] != "mhr":
            continue
        nm = st["name"].replace("skel_", "")
        key = "_".join(nm.split("_")[:2]) if nm.startswith(("hand", "foot")) else nm
        d = depth(st["verts"], exact=True); a0 = agg.setdefault(key, [0, 0]); a0[0] += len(d); a0[1] += int((d < 0).sum())
    print("  outside the rest skin after seating:", ", ".join(f"{k} {100 * o / n:.0f}%" for k, (n, o) in sorted(agg.items())))


if __name__ == "__main__":
    main()
