"""Joint-rule tests: pins, range of motion, shrug, respiration, and browser (JS) parity. Needs the shipped atlas."""
import json, os, shutil, subprocess
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from anatomy.runtime import Atlas, kabsch

ATLAS = os.environ.get("ATLAS_PREFIX", os.path.join(os.path.dirname(__file__), "..", "atlas", "atlas"))
pytestmark = pytest.mark.skipif(not os.path.exists(ATLAS + ".json"), reason="no built atlas found")


@pytest.fixture(scope="module")
def atlas():
    a = Atlas(ATLAS)
    if not a.meta.get("joints"):
        pytest.skip("atlas has no joint table (rebuild with anatomy.atlas_build)")
    return a


def skin_with(a, motions):
    """Skin = rest everywhere; the support vertices of the named segments get an exact rigid motion."""
    skin = np.zeros((a.meta["n_skin_vertices"], 3))
    for sg in a.segments:
        skin[sg["idx"]] = sg["rest"]
    for sg in a.segments:
        if sg["name"] in motions:
            rv, t = motions[sg["name"]]
            c = sg["rest"].mean(0)
            skin[sg["idx"]] = (sg["rest"] - c) @ Rot.from_rotvec(rv).as_matrix().T + c + t
    return skin


BIG = {"shoulder_l": ([0, 0, 0], [0, -0.06, 0]), "uparm_l": ([0.3, 0.2, 1.6], [0, -0.04, 0]),
       "uparm_r": ([0.0, -0.4, -1.2], [0, 0, 0]), "thigh_r": ([1.2, 0, 0], [0, 0, 0.02]),
       "thigh_l": ([-0.8, 0.2, 0], [0, 0, 0]), "head": ([0.5, 0.8, 0.3], [0.02, -0.01, 0]),
       "thorax_up": ([0, 0.3, 0.1], [0, 0, 0.01]), "thorax_low": ([0, 0.15, 0.05], [0, 0, 0])}


def gap(a, j, pos, posed):
    by = {s["name"]: s for s in a.structures}
    J = np.asarray(j["center"]); c = by[j["child"]]["pos"]
    sel = np.linalg.norm(c - J, axis=1) < 0.035
    if sel.sum() < 5:
        sel = np.argsort(np.linalg.norm(c - J, axis=1))[:30]
    return cKDTree(posed[j["parent"]]).query(posed[j["child"]][sel])[0].min()


def test_identity_pose_is_rest_for_every_structure(atlas):
    out = atlas.pose(skin_with(atlas, {}))
    assert max(np.abs(out[s["name"]] - s["pos"]).max() for s in atlas.structures) < 1e-4


def test_pinned_joints_hold_under_large_motions(atlas):
    rest = {s["name"]: s["pos"] for s in atlas.structures}
    posed = atlas.pose(skin_with(atlas, BIG))
    tol = {"ball": 0.0025, "condyloid": 0.0015, "saddle": 0.006, "plane": 0.006}
    for j in atlas.meta["joints"]:
        if j["type"] == "costovertebral":
            continue
        d = abs(gap(atlas, j, rest, posed) - gap(atlas, j, rest, rest))
        assert d < tol[j["type"]], (j["name"], d)       # glenohumeral/hip/skull sockets stay sockets


def test_range_of_motion_clamps_the_skull(atlas):
    posed_skin = skin_with(atlas, {"head": ([0, np.radians(170), 0], [0, 0, 0])})
    T, cf, sf, F = atlas.frames(posed_skin)
    c1 = cf[atlas.meta["chain"].index("vertebrae_C1")]
    rel = Rot.from_matrix(c1[1].T @ F["skull"][1]).magnitude()
    limit = [j for j in atlas.meta["joints"] if j["name"] == "ao"][0]["limit_deg"]
    assert np.degrees(rel) <= limit + 0.5, np.degrees(rel)


def test_shrug_lifts_clavicle_scapula_and_humeral_head_together(atlas):
    by = {s["name"]: s["pos"] for s in atlas.structures}
    lift = 0.05                                                    # shoulder cap rises 5 cm (y is down)
    posed = atlas.pose(skin_with(atlas, {"shoulder_l": ([0, 0, 0], [0, -lift, 0])}))
    rise = lambda n: (by[n][:, 1].mean() - posed[n][:, 1].mean())
    assert rise("clavicula_left") > 0.008 and rise("scapula_left") > 0.008
    assert abs(rise("humerus_left") - rise("scapula_left")) < 0.004   # the ball rides with its socket
    assert abs(rise("scapula_right")) < 1e-3                         # the other side does not move
    joints = {j["name"]: j for j in atlas.meta["joints"]}
    rest = {s["name"]: s["pos"] for s in atlas.structures}
    d = abs(gap(atlas, joints["gh_l"], rest, posed) - gap(atlas, joints["gh_l"], rest, rest))
    assert d < 0.0015, d                                            # was ~30 mm before joint pinning


def test_brain_moves_rigidly_with_the_skull(atlas):
    posed = atlas.pose(skin_with(atlas, BIG))
    by = {s["name"]: s["pos"] for s in atlas.structures}
    for n in ("skull", "brain"):
        if n not in by:
            pytest.skip("no " + n)
    R1, _, _ = kabsch(by["skull"], posed["skull"]); R2, _, _ = kabsch(by["brain"], posed["brain"])
    assert np.degrees(Rot.from_matrix(R1.T @ R2).magnitude()) < 0.1


def test_respiration_keeps_ribs_on_their_vertebrae_and_lifts_the_sternum(atlas):
    skin = skin_with(atlas, {})
    rest = atlas.pose(skin, breath=0.0)
    insp = atlas.pose(skin, breath=1.0)
    joints = {j["name"]: j for j in atlas.meta["joints"]}
    T, cf, sf, F = atlas.frames(skin, 1.0)
    chain = atlas.meta["chain"]
    for k in (3, 6, 9):
        j = joints[f"cv_rib_left_{k}"]
        fv = cf[chain.index(f"vertebrae_T{k}")]
        Jc = np.asarray(j["center"]); Jlive = (Jc - fv[2]) @ fv[1].T + fv[3]
        near = np.linalg.norm(atlas.structures[[s["name"] for s in atlas.structures].index(j["child"])]["pos"] - Jc, axis=1) < 0.006
        d = np.linalg.norm(insp[j["child"]][near].mean(0) - Jlive)
        assert d < 0.004, (k, d)                                  # the costovertebral joint stays closed (4 mm: the ribs are 7-10% larger since tools/fit_skeleton_to_mhr.py; it was 3 mm)
    up = lambda n: rest[n][:, 1].mean() - insp[n][:, 1].mean()   # y is down: positive = rises
    fwd = lambda n: rest[n][:, 2].mean() - insp[n][:, 2].mean()  # front is -z: positive = moves anterior
    assert 0.001 < up("sternum") < 0.02 and fwd("sternum") > 0.0005
    assert up("rib_left_6") > 0.0005


def test_js_runtime_matches_numpy_with_joints_and_breathing(atlas, tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    skin = skin_with(atlas, BIG)
    breath, depth = 0.8, 1.5
    out = atlas.pose(skin, breath=breath, depth=depth)
    exp = np.concatenate([out[s["name"]].reshape(-1) for s in atlas.structures]).astype("<f4")
    sk, ep = tmp_path / "skin.bin", tmp_path / "exp.bin"
    skin.astype("<f4").tofile(sk); exp.tofile(ep)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"skin": str(sk), "breath": breath, "depth": depth, "expected": str(ep)}))
    here = os.path.dirname(__file__)
    r = subprocess.run(["node", os.path.join(here, "js", "parity_pose.mjs"), ATLAS, str(case)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_js_runtime_matches_numpy_with_the_physiologic_spine(atlas, tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    skin = skin_with(atlas, BIG)
    out = atlas.pose(skin, breath=0.5, depth=1.0, spine="physio")
    exp = np.concatenate([out[s["name"]].reshape(-1) for s in atlas.structures]).astype("<f4")
    sk, ep = tmp_path / "skin.bin", tmp_path / "exp.bin"
    skin.astype("<f4").tofile(sk); exp.tofile(ep)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"skin": str(sk), "breath": 0.5, "depth": 1.0, "spine": "physio", "expected": str(ep)}))
    r = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "js", "parity_pose.mjs"), ATLAS, str(case)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def _head_in_socket(atlas, posed, side):
    from anatomy.atlas_build import joint_head_center
    by = {s["name"]: s["pos"] for s in atlas.structures}
    hc = lambda X: joint_head_center(X * 1000.0, -1)[0] / 1000.0
    d0 = cKDTree(by[f"scapula_{side}"]).query(hc(by[f"humerus_{side}"]))[0]
    d1 = cKDTree(posed[f"scapula_{side}"]).query(hc(posed[f"humerus_{side}"]))[0]
    return 1000 * (d1 - d0)


def test_without_joint_rules_the_shrug_drives_the_ball_into_the_scapula(atlas):
    """Regression for the project's stage before joint rules: same shrug, with and without the joint model."""
    skin = skin_with(atlas, {"shoulder_l": ([0, 0, 0], [0, -0.05, 0]), "uparm_l": ([0, 0, 0], [0, -0.05, 0])})
    legacy = abs(_head_in_socket(atlas, atlas.pose(skin, joints=False), "left"))
    joint = abs(_head_in_socket(atlas, atlas.pose(skin), "left"))
    assert legacy > 5.0 and joint < 2.5, (legacy, joint)


def test_physiologic_spine_split_follows_in_vivo_per_level_data(atlas):
    from anatomy.runtime import _AXIAL
    twist = np.radians(20.0)
    skin = skin_with(atlas, {"thorax_up": ([0, twist, 0], [0, 0, 0]), "thorax_low": ([0, twist, 0], [0, 0, 0])})
    by = {s["name"]: s for s in atlas.structures}
    names = [n for n in atlas.meta["chain"] if n.replace("vertebrae_", "") in _AXIAL]       # L5 .. T1

    def profile(posed):
        psi = np.array([np.degrees(Rot.from_matrix(kabsch(by[n]["pos"], posed[n])[0]).as_rotvec()[1]) for n in names])
        return psi, np.diff(psi)
    psi, inc = profile(atlas.pose(skin, spine="physio"))
    ref = np.array([_AXIAL[n.replace("vertebrae_", "")] for n in names[1:]])
    assert np.corrcoef(np.abs(inc), ref)[0, 1] > 0.98                         # matches the measured per-level profile
    assert abs(abs(psi[-1] - psi[0]) - 20.0) < 2.0                            # T1 still carries the whole measured trunk rotation
    lumbar_share = abs(psi[names.index("vertebrae_L1")] - psi[0]) / abs(psi[-1] - psi[0])
    assert 0.2 < lumbar_share < 0.35                                          # lumbar share ~28% (in-vivo CT), not the skin-block split
    psi0, inc0 = profile(atlas.pose(skin))                                    # default mode keeps the skin-driven split
    assert abs(np.corrcoef(np.abs(inc0), ref)[0, 1]) < 0.95


def test_costal_cartilage_stays_joined_to_ribs_and_sternum_under_large_motion(atlas):
    """Cartilage is a flexible link: its ends follow the ribs and the sternum, so neither junction opens."""
    by = {s["name"]: s["pos"] for s in atlas.structures}
    rest_gap = cKDTree(by["sternum"]).query(by["costal_cartilages"])[0].min()
    posed = atlas.pose(skin_with(atlas, BIG), breath=1.0)
    gap_st = cKDTree(posed["sternum"]).query(posed["costal_cartilages"])[0].min()
    assert gap_st <= rest_gap + 0.004
    for r in range(3, 9):
        n = f"rib_left_{r}"
        g0 = cKDTree(by[n]).query(by["costal_cartilages"])[0].min()
        g1 = cKDTree(posed[n]).query(posed["costal_cartilages"])[0].min()
        assert g1 <= g0 + 0.004, (n, g0, g1)


def test_scapula_keeps_contact_with_the_rib_cage_when_the_arm_is_raised_forward(atlas):
    """Scapulothoracic glide: the scapula stays near the thorax instead of floating off it (it floated 16 mm away without the contact step)."""
    by = {s["name"]: s["pos"] for s in atlas.structures}
    ribs = np.vstack([by[n] for n in by if n.startswith("rib_left_") and 2 <= int(n.split("_")[-1]) <= 9])
    g0 = cKDTree(ribs).query(by["scapula_left"])[0].min()
    for deg in (30, 90, 150):
        P = atlas.pose(skin_with(atlas, {"uparm_l": (np.array([1.0, 0, 0]) * np.radians(deg), [0, 0, 0])}))
        rib_p = np.vstack([P[n] for n in P if n.startswith("rib_left_") and 2 <= int(n.split("_")[-1]) <= 9])
        g = cKDTree(rib_p).query(P["scapula_left"])[0].min()
        # 7.5 mm: the scapula is about 17% larger since it was sized to the MHR shoulder (tools/seat_limbs_on_mhr.py); it was 6 mm
        assert abs(g - g0) < 0.0075, (deg, g0, g)


# ---- rig driven by the MHR skeleton (atlas.json["mhr"], tools/bind_mhr_joints.py) ----

def mhr_joints_with(a, motions, lengthen=None):
    """Synthetic MHR skeleton: every joint at rest; the joint bound to each named segment gets a rigid motion, and
    `lengthen` {bone: factor} moves a limb's child joint to change that bone's length."""
    m = a.meta["mhr"]; n = m["n_joints"]
    pos = np.zeros((n, 3)); rot = np.tile(np.eye(3), (n, 1, 1))
    for j, p in m["rest_pos"].items():
        pos[int(j)] = p; rot[int(j)] = np.asarray(m["rest_rot"][j]).reshape(3, 3)
    for seg, (rv, t) in motions.items():
        j = m["segments"][seg]
        rot[j] = Rot.from_rotvec(rv).as_matrix() @ rot[j]; pos[j] = pos[j] + np.asarray(t)
    for bone, f in (lengthen or {}).items():
        lb = m["limbs"][bone]
        pos[lb["child"]] = pos[lb["joint"]] + f * (np.asarray(m["rest_pos"][str(lb["child"])]) - np.asarray(m["rest_pos"][str(lb["joint"])]))
    return pos, rot


def test_mhr_joints_at_rest_reproduce_the_rest_pose(atlas):
    if not atlas.meta.get("mhr"):
        pytest.skip("atlas has no MHR joint binding")
    skin = skin_with(atlas, {})
    out = atlas.pose(skin, mhr_joints=mhr_joints_with(atlas, {}))
    for s in atlas.structures:
        assert np.abs(out[s["name"]] - s["pos"]).max() < 1e-4, s["name"]


def test_mhr_joint_drives_its_segment_and_scales_the_limb_bone(atlas):
    if not atlas.meta.get("mhr"):
        pytest.skip("atlas has no MHR joint binding")
    skin = skin_with(atlas, {})                                # the skin does not move: only the joints do
    rv = np.array([0.0, 0.0, 0.7])
    T, cf, sf, F = atlas.frames(skin, mhr_joints=mhr_joints_with(atlas, {"uparm_l": (rv, [0, 0, 0])}, lengthen={"humerus_left": 0.9}))
    by = {s["name"]: s for s in atlas.structures}
    assert Rot.from_matrix(T[[g["name"] for g in atlas.segments].index("uparm_l")][1]).magnitude() == pytest.approx(0.7, abs=1e-6)
    assert F["humerus_left"][4] == pytest.approx(0.9, abs=1e-4)          # bone length follows the joints
    posed = atlas.pose(skin, mhr_joints=mhr_joints_with(atlas, {"uparm_l": (rv, [0, 0, 0])}, lengthen={"humerus_left": 0.9}))
    h = by["humerus_left"]["pos"]; span = lambda p: np.linalg.norm(p - p.mean(0), axis=1).max()
    assert span(posed["humerus_left"]) == pytest.approx(0.9 * span(h), rel=1e-4)
    # defined on the model's skeleton: the humeral head and the scapula's glenoid both sit on the live MHR shoulder joint
    pos, rot = mhr_joints_with(atlas, {"uparm_l": (rv, [0, 0, 0])})
    pos[atlas.meta["mhr"]["limbs"]["humerus_left"]["joint"]] += [0.03, -0.02, 0.01]          # the shoulder moves (a shrug)
    F2 = atlas.frames(skin, mhr_joints=(pos, rot))[3]
    live = pos[atlas.meta["mhr"]["limbs"]["humerus_left"]["joint"]]
    gh = np.asarray({j["name"]: j for j in atlas.meta["joints"]}["gh_l"]["center"])
    if not atlas.meta["mhr"].get("limb_seated"):
        return      # an atlas aligned to its CT keeps the humeri where the scan has them: only seated atlases pin them to the joint
    for bone in ("humerus_left", "scapula_left"):
        f = F2[bone]
        assert np.linalg.norm((gh - f[2]) @ f[1].T * f[4] + f[3] - live) < 1e-9, bone
    J = {j["name"]: j for j in atlas.meta["joints"]}
    ac = np.asarray(J["ac_l"]["center"]); ap = lambda f, x: (x - f[2]) @ f[1].T * f[4] + f[3]
    assert np.linalg.norm(ap(F2["clavicula_left"], ac) - ap(F2["scapula_left"], ac)) < 0.004     # clavicle still meets the scapula


def test_js_runtime_matches_numpy_when_driven_by_mhr_joints(atlas, tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    if not atlas.meta.get("mhr"):
        pytest.skip("atlas has no MHR joint binding")
    skin = skin_with(atlas, BIG)
    pos, rot = mhr_joints_with(atlas, BIG, lengthen={"humerus_left": 1.12, "humerus_right": 0.9, "femur_left": 1.05})
    out = atlas.pose(skin, breath=0.4, depth=1.0, spine="physio", mhr_joints=(pos, rot))
    exp = np.concatenate([out[s["name"]].reshape(-1) for s in atlas.structures]).astype("<f4")
    sk, ep, jp = tmp_path / "skin.bin", tmp_path / "exp.bin", tmp_path / "joints.bin"
    skin.astype("<f4").tofile(sk); exp.tofile(ep); np.concatenate([pos.reshape(-1), rot.reshape(-1)]).astype("<f4").tofile(jp)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"skin": str(sk), "breath": 0.4, "depth": 1.0, "spine": "physio", "expected": str(ep), "joints": str(jp)}))
    here = os.path.dirname(__file__)
    r = subprocess.run(["node", os.path.join(here, "js", "parity_pose.mjs"), ATLAS, str(case)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_js_runtime_matches_numpy_with_body_scaling(atlas, tmp_path):
    """A body bigger or smaller than the neutral one: the patches are scaled as well as moved, with and without MHR joints."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    skin = skin_with(atlas, BIG)
    sizes = {"pelvis": 0.8, "lumbar": 0.78, "thorax_low": 0.75, "thorax_up": 0.82, "head": 1.07, "uparm_l": 0.9, "uparm_r": 1.1,
             "thigh_l": 1.15, "thigh_r": 0.9, "shoulder_l": 0.85, "shoulder_r": 1.12}
    for sg in atlas.segments:
        c = skin[sg["idx"]].mean(0)
        skin[sg["idx"]] = c + sizes[sg["name"]] * (skin[sg["idx"]] - c)
    T = atlas.segment_transforms(skin)
    for f, sg in zip(T, atlas.segments):                       # neighbouring patches share a few vertices, so not exact
        assert f[4] == pytest.approx(sizes[sg["name"]], abs=0.03), sg["name"]
    here = os.path.dirname(__file__)
    cases = [("skin", None)]
    if atlas.meta.get("mhr"):
        cases.append(("joints", mhr_joints_with(atlas, BIG, lengthen={"humerus_left": 1.12, "femur_right": 0.95})))
    for tag, mj in cases:
        out = atlas.pose(skin, breath=0.6, depth=1.0, spine="physio", mhr_joints=mj)
        exp = np.concatenate([out[s["name"]].reshape(-1) for s in atlas.structures]).astype("<f4")
        sk, ep = tmp_path / f"skin_{tag}.bin", tmp_path / f"exp_{tag}.bin"
        skin.astype("<f4").tofile(sk); exp.tofile(ep)
        case = {"skin": str(sk), "breath": 0.6, "depth": 1.0, "spine": "physio", "expected": str(ep)}
        if mj is not None:
            jp = tmp_path / "joints_scaled.bin"
            np.concatenate([mj[0].reshape(-1), mj[1].reshape(-1)]).astype("<f4").tofile(jp); case["joints"] = str(jp)
        cp = tmp_path / f"case_{tag}.json"; cp.write_text(json.dumps(case))
        r = subprocess.run(["node", os.path.join(here, "js", "parity_pose.mjs"), ATLAS, str(cp)], capture_output=True, text=True)
        assert r.returncode == 0, tag + ": " + r.stdout + r.stderr


def test_body_scaling_keeps_joints_pinned_and_sizes_the_ribcage(atlas):
    skin = skin_with(atlas, {})
    for sg in atlas.segments:                                   # a body 20% smaller everywhere, in the rest pose
        skin[sg["idx"]] = 0.8 * sg["rest"]
    posed = atlas.pose(skin)
    T, cf, sf, F = atlas.frames(skin)
    by = {s["name"]: s for s in atlas.structures}
    width = lambda P: np.ptp(np.vstack([P[n] for n in P if n.startswith("rib_")])[:, 0])
    assert width(posed) == pytest.approx(0.8 * width({n: s["pos"] for n, s in by.items()}), rel=0.03)
    J = {j["name"]: j for j in atlas.meta["joints"]}
    for name in ("gh_l", "gh_r", "hip_l", "hip_r"):             # ball still in its socket after scaling
        j = J[name]; c = np.asarray(j["center"]); par = F.get(j["parent"])
        if par is None:
            continue
        live = (c - par[2]) @ par[1].T * par[4] + par[3]
        child = F[j["child"]]
        assert np.linalg.norm(child[3] - live) < 1e-9, name
