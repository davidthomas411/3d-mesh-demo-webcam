"""Atlas tests. Part 1 is synthetic (always runs). Part 2 needs a built atlas (ATLAS_PREFIX env)."""
import os
import numpy as np
import pytest

from anatomy.atlas_build import umeyama, joint_head_center, enforce_containment
from anatomy.runtime import weighted_kabsch
from anatomy.classes import group_of
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(8, 3))
    R = Rot.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()
    dst = 1.37 * src @ R.T + np.array([0.3, -2, 5])
    s, R2, t = umeyama(src, dst)
    assert abs(s - 1.37) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [0.3, -2, 5])


def test_sphere_ransac_finds_femoral_head():
    rng = np.random.default_rng(1)
    c, r = np.array([10.0, -300.0, 4.0]), 24.0       # y is down: head is the most superior (-y) part
    u = rng.normal(size=(1500, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    head = c + r * u
    shaft = np.column_stack([rng.normal(0, 12, 1500), rng.uniform(-260, -100, 1500), rng.normal(0, 12, 1500)])
    est, rad, frac = joint_head_center(np.vstack([head[head[:, 1] < c[1] + 10], shaft]), -1)
    assert np.linalg.norm(est - c) < 3 and abs(rad - r) < 3


def test_weighted_kabsch_exact_and_robust_to_weights():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3))
    R = Rot.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    dst = src @ R.T + np.array([1.0, 2.0, 3.0])
    w = rng.uniform(0.1, 1, 40)
    Rk, cs, cd = weighted_kabsch(src, dst, w)
    assert np.allclose(Rk, R, atol=1e-9) and np.allclose(cd - Rk @ cs, [1, 2, 3], atol=1e-9)


def test_group_table_covers_ts_names():
    for n, g in [("vertebrae_L3", "spine"), ("rib_left_7", "ribcage"), ("aorta", "artery"),
                 ("portal_vein_and_splenic_vein", "vein"), ("lung_lower_lobe_left", "lung"),
                 ("gluteus_maximus_left", "muscle"), ("femur_right", "pelvis_leg"), ("liver", "gi"),
                 ("kidney_left", "urinary"), ("skull", "skull"), ("heart_myocardium", "heart")]:
        assert group_of(n) == g, (n, group_of(n))


ATLAS = os.environ.get("ATLAS_PREFIX", os.path.join(os.path.dirname(__file__), "..", "atlas", "atlas"))
needs_atlas = pytest.mark.skipif(not os.path.exists(ATLAS + ".json"), reason="no built atlas found")


@needs_atlas
def test_built_atlas_is_total_segmentator_scale():
    from anatomy.runtime import Atlas
    a = Atlas(ATLAS)
    names = a.names()
    assert len(names) >= 100
    assert {"liver", "aorta", "spleen", "stomach", "heart", "skull"} <= set(names)
    assert a.meta["version"] == 2 and len(a.meta["chain"]) >= 20
    assert max(len(s["faces"]) for s in a.structures) < 40000


def _synthetic_pose(a, motions):
    """Skin = rest everywhere except the support vertices of the named segments, which get a known
    rigid motion {segment: (rotvec, translation)}. Exact segment motions, so the rig is tested alone."""
    skin = np.zeros((a.meta["n_skin_vertices"], 3))
    for sg in a.segments:
        skin[sg["idx"]] = sg["rest"]
    for sg in a.segments:
        if sg["name"] in motions:
            rv, t = motions[sg["name"]]
            c = sg["rest"].mean(0)
            skin[sg["idx"]] = (sg["rest"] - c) @ Rot.from_rotvec(rv).as_matrix().T + c + t
    return skin


@needs_atlas
def test_identity_and_global_rigid_motion_move_everything_exactly():
    from anatomy.runtime import Atlas
    a = Atlas(ATLAS)
    ident = a.pose(_synthetic_pose(a, {}))
    assert max(np.abs(ident[s["name"]] - s["pos"]).max() for s in a.structures) < 1e-4
    R = Rot.from_rotvec([0.2, -0.4, 0.3]).as_matrix(); t = np.array([0.3, 0.1, -0.5])
    skin = _synthetic_pose(a, {})
    for sg in a.segments:
        skin[sg["idx"]] = sg["rest"] @ R.T + t
    moved = a.pose(skin)
    # everything (bones AND soft tissue) must follow a whole-body rigid motion
    assert max(np.abs(moved[s["name"]] - (s["pos"] @ R.T + t)).max() for s in a.structures) < 2e-3


@needs_atlas
def test_bones_stay_rigid_under_articulation():
    from anatomy.runtime import Atlas
    a = Atlas(ATLAS)
    skin = _synthetic_pose(a, {"head": ([0.3, 0.2, 0.4], [0.02, 0.01, 0]),
                               "thorax_up": ([0.0, 0.25, 0.1], [0, 0, 0.01]),
                               "thorax_low": ([0.0, 0.12, 0.05], [0, 0, 0]),
                               "uparm_l": ([0.0, 0.0, 0.8], [0, 0.02, 0]),
                               "thigh_r": ([0.5, 0, 0], [0, 0, 0.01])})
    posed = a.pose(skin)
    worst = 0.0
    for s in a.structures:
        if s["bind"] == "soft":
            continue
        p, q = s["pos"][::9], posed[s["name"]][::9]
        dp = np.linalg.norm(p[:, None] - p[None], axis=-1); dq = np.linalg.norm(q[:, None] - q[None], axis=-1)
        worst = max(worst, np.abs(dp - dq).max())
    assert worst < 1e-4, worst        # bones do not deform at all (float32 storage only)


@needs_atlas
def test_spine_twist_distributes_monotonically_and_keeps_spacing():
    from anatomy.runtime import Atlas, kabsch
    a = Atlas(ATLAS)
    twist = 0.5                                        # rad about the vertical (y) axis
    skin = _synthetic_pose(a, {"thorax_up": ([0, twist, 0], [0, 0, 0]), "thorax_low": ([0, twist * 0.5, 0], [0, 0, 0]),
                               "head": ([0, twist * 1.2, 0], [0, 0, 0])})
    posed = a.pose(skin)
    by = {s["name"]: s for s in a.structures}
    ang = []
    for n in a.meta["chain"]:
        R, _, _ = kabsch(by[n]["pos"], posed[n])
        ang.append(Rot.from_matrix(R).as_rotvec()[1])
    ang = np.array(ang)                                # bottom (L5) -> top (C1)
    assert ang[0] < 0.1 and ang[-1] > 0.45 * twist     # pelvis end ~fixed, head end carries the twist
    assert np.all(np.diff(ang) > -0.02), np.round(ang, 2)       # monotone: no counter-rotation, no kinks
    cen = lambda P, n: P[n].mean(0)
    for u, v in zip(a.meta["chain"][:-1], a.meta["chain"][1:]):
        d0 = np.linalg.norm(cen({u: by[u]["pos"], v: by[v]["pos"]}, u) - cen({u: by[u]["pos"], v: by[v]["pos"]}, v))
        d1 = np.linalg.norm(cen(posed, u) - cen(posed, v))
        assert abs(d1 - d0) < 0.004, (u, v, d0, d1)    # vertebra spacing preserved within 4 mm


@needs_atlas
def test_ribs_keep_their_vertebra():
    from anatomy.runtime import Atlas
    a = Atlas(ATLAS)
    skin = _synthetic_pose(a, {"thorax_up": ([0.3, 0.4, 0.2], [0.01, 0, 0]), "thorax_low": ([0.1, 0.2, 0.1], [0, 0, 0])})
    posed = a.pose(skin)
    by = {s["name"]: s for s in a.structures}
    for k in range(1, 13):
        r, v = f"rib_left_{k}", f"vertebrae_T{k}"
        if r in by and v in by:
            d0 = np.linalg.norm(by[r]["pos"].mean(0) - by[v]["pos"].mean(0))
            d1 = np.linalg.norm(posed[r].mean(0) - posed[v].mean(0))
            assert abs(d1 - d0) < 1e-4, (r, d0, d1)      # costovertebral joint: exactly rigid


def test_chain_weights_partition_of_unity_and_ordering():
    from anatomy import rig
    lv = {"pelvis": -0.95, "lumbar": -1.10, "thorax_low": -1.25, "thorax_up": -1.40, "C7": -1.474, "C2": -1.525, "hip_y": -0.906}
    y = np.linspace(-1.7, -0.8, 200)
    W = rig.chain_weights_y(y, lv)
    assert np.allclose(W.sum(1), 1) and (W >= -1e-12).all()
    assert W[0].argmax() == 4 and W[-1].argmax() == 0               # head at the top, pelvis at the bottom
    assert rig.attach_for("rib_left_7", ["vertebrae_T6", "vertebrae_T7"]) == [(1, 1.0)]
    assert rig.classify("vertebrae_S1") == ("bone", "pelvis") and rig.classify("liver")[0] == "soft"


def test_enforce_containment_pushes_structures_inside_the_skin():
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
    rng = np.random.default_rng(0)
    p = rng.normal(size=(2000, 3)); p = p / np.linalg.norm(p, axis=1, keepdims=True) * rng.uniform(0.0, 0.215, (2000, 1))
    st = [{"name": "rib_left_5", "verts": p.copy(), "faces": np.zeros((0, 3), int)}]     # 'ribcage': needs >= 10 mm depth
    enforce_containment(st, np.asarray(sph.vertices), np.asarray(sph.faces), verbose=False)
    r = np.linalg.norm(st[0]["verts"], axis=1)
    assert (r < 0.2).mean() > 0.97 and r.max() < 0.205              # was 10% outside; now (almost) all inside


def test_js_matches_numpy(tmp_path):
    import json, shutil, subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    rng = np.random.default_rng(5)
    live = rng.normal(size=(500, 3)).astype(np.float32)
    idx = rng.choice(500, 60, replace=False)
    w = rng.uniform(0.2, 1, 60).astype(np.float32); w /= w.sum()
    Rt = Rot.from_rotvec([0.4, -0.7, 0.2]).as_matrix()
    s_rest = (live[idx] - 0.1) @ Rt.T + 0.3 + rng.normal(size=(60, 3)).astype(np.float32) * 0.01
    R, cs, cd = weighted_kabsch(s_rest.astype(np.float64), live[idx].astype(np.float64), w.astype(np.float64))
    case = {"sRest": s_rest.tolist(), "sIdx": idx.tolist(), "sW": w.tolist(), "live": live.tolist(), "R": R.reshape(-1).tolist()}
    p = tmp_path / "case.json"; p.write_text(json.dumps(case))
    r = subprocess.run(["node", "tests/js/parity.mjs", str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
