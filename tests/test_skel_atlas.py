"""SKEL skeleton atlas (atlas/skel.*, built locally by tools/build_skel_atlas.py; not in the repository): bones rigid with MHR
joints, the jaw hinge, and browser (JS) parity. Skipped when the atlas is not there."""
import json, os, shutil, subprocess
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from anatomy.runtime import Atlas

SKEL = os.path.join(os.path.dirname(__file__), "..", "atlas", "skel")
pytestmark = pytest.mark.skipif(not os.path.exists(SKEL + ".json"), reason="no SKEL atlas on this machine")


@pytest.fixture(scope="module")
def atlas():
    return Atlas(SKEL)


def rest_skin(a):
    skin = np.zeros((a.meta["n_skin_vertices"], 3))
    for sg in a.segments:
        skin[sg["idx"]] = sg["rest"]
    cr = a.meta["mhr"].get("cranium")
    if cr:
        skin[np.asarray(cr["idx"])] = np.asarray(cr["rest"])
    for st in a.structures:                                          # the skin samples that measure limb girth
        b = st.get("mhr") or {}
        if "girth_idx" in b:
            skin[np.asarray(b["girth_idx"])] = np.asarray(b["girth_pos"])
    return skin


def rest_joints(a):
    m = a.meta["mhr"]; n = m["n_joints"]
    pos = np.zeros((n, 3)); rot = np.tile(np.eye(3), (n, 1, 1))
    for j, p in m["rest_pos"].items():
        pos[int(j)] = p; rot[int(j)] = np.asarray(m["rest_rot"][j]).reshape(3, 3)
    return pos, rot


def test_rest_pose_is_reproduced(atlas):
    out = atlas.pose(rest_skin(atlas), mhr_joints=rest_joints(atlas))
    for s in atlas.structures:
        assert np.abs(out[s["name"]] - s["pos"]).max() < 1e-4, s["name"]


def test_limb_bone_follows_its_joint_with_length_from_the_joints_and_thickness_from_the_skin(atlas):
    by = {s["name"]: s for s in atlas.structures}
    b = by["skel_ulna_l"]["mhr"]
    m = atlas.meta["mhr"]
    rest_j, rest_c = np.asarray(m["rest_pos"][str(b["joint"])]), np.asarray(m["rest_pos"][str(b["child"])])
    ax = (rest_c - rest_j) / np.linalg.norm(rest_c - rest_j)
    pos, rot = rest_joints(atlas)
    Rj = Rot.from_rotvec([0.2, -0.5, 0.9]).as_matrix()
    rot[b["joint"]] = Rj @ rot[b["joint"]]
    pos[b["joint"]] += [0.05, -0.02, 0.03]
    pos[b["child"]] = pos[b["joint"]] + 1.1 * Rj @ (rest_c - rest_j)            # the forearm is 10% longer
    skin = rest_skin(atlas)
    g = np.asarray(b["girth_pos"]) - rest_j                                      # and 20% thinner
    g = 0.8 * g + 0.2 * np.outer(g @ ax, ax)
    skin[np.asarray(b["girth_idx"])] = g @ Rj.T + pos[b["joint"]]
    out = atlas.pose(skin, mhr_joints=(pos, rot))
    x = by["skel_ulna_l"]["pos"] - rest_j
    along = np.outer(x @ ax, ax)
    expect = (0.8 * (x - along) + 1.1 * along) @ Rj.T + pos[b["joint"]]
    assert np.abs(out["skel_ulna_l"] - expect).max() < 2e-4
    assert np.abs(out["skel_ulna_r"] - by["skel_ulna_r"]["pos"]).max() < 1e-4   # the other arm did not move


def test_jaw_hinge_opens_the_chin_downward_and_leaves_the_skull(atlas):
    by = {s["name"]: s for s in atlas.structures}
    skin, joints = rest_skin(atlas), rest_joints(atlas)
    closed = atlas.pose(skin, mhr_joints=joints)
    opened = atlas.pose(skin, mhr_joints=joints, jaw_deg=20.0)
    assert np.abs(opened["skel_skull"] - closed["skel_skull"]).max() < 1e-9
    jaw = by["skel_jaw"]; pivot = np.asarray(jaw["jaw"]["pivot"])
    chin = np.argmin(jaw["pos"][:, 2])                                # most anterior point of the jaw (front is -z)
    move = opened["skel_jaw"][chin] - closed["skel_jaw"][chin]
    assert move[1] > 0.01 and move[2] > 0                             # y is down: the chin drops, and swings back
    r = np.linalg.norm((jaw["pos"][chin] - pivot)[1:])               # distance from the hinge line (about x)
    assert np.linalg.norm(move) == pytest.approx(2 * r * np.sin(np.radians(10)), rel=0.05)
    near = np.linalg.norm(jaw["pos"] - pivot, axis=1) < 0.06          # points near the hinge barely move
    assert np.linalg.norm(opened["skel_jaw"] - closed["skel_jaw"], axis=1)[near].max() < 0.03


def test_js_runtime_matches_numpy_for_the_skel_atlas(atlas, tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    skin = rest_skin(atlas)
    for sg in atlas.segments:                                        # move and resize a few segments
        if sg["name"] in ("head", "thorax_up", "pelvis"):
            c = sg["rest"].mean(0)
            skin[sg["idx"]] = 1.06 * (sg["rest"] - c) @ Rot.from_rotvec([0.1, 0.3, -0.2]).as_matrix().T + c + [0.01, 0.0, 0.02]
    pos, rot = rest_joints(atlas)
    rng = np.random.default_rng(3)
    for j in atlas.meta["mhr"]["rest_pos"]:
        rot[int(j)] = Rot.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix() @ rot[int(j)]
        pos[int(j)] += rng.normal(0, 0.02, 3)
    out = atlas.pose(skin, spine="physio", mhr_joints=(pos, rot), jaw_deg=17.0)
    exp = np.concatenate([out[s["name"]].reshape(-1) for s in atlas.structures]).astype("<f4")
    sk, ep, jp = tmp_path / "skin.bin", tmp_path / "exp.bin", tmp_path / "joints.bin"
    skin.astype("<f4").tofile(sk); exp.tofile(ep); np.concatenate([pos.reshape(-1), rot.reshape(-1)]).astype("<f4").tofile(jp)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"skin": str(sk), "breath": 0.0, "depth": 1.0, "spine": "physio", "expected": str(ep), "joints": str(jp), "jawDeg": 17.0}))
    here = os.path.dirname(__file__)
    r = subprocess.run(["node", os.path.join(here, "js", "parity_pose.mjs"), SKEL, str(case)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
