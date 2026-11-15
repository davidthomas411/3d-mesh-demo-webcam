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
    tol = {"ball": 0.0015, "condyloid": 0.0015, "saddle": 0.006, "plane": 0.006}
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
        assert d < 0.003, (k, d)                                  # the costovertebral joint stays closed
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
