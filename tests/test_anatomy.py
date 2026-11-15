"""Atlas tests. Part 1 is synthetic (always runs). Part 2 needs a built atlas (ATLAS_PREFIX env)."""
import os
import numpy as np
import pytest

from anatomy.atlas_build import umeyama, joint_head_center, bind_structure
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


def test_binding_follows_rigid_motion():
    rng = np.random.default_rng(3)
    skin = rng.normal(size=(3000, 3)) * 0.2
    skin = skin / np.linalg.norm(skin, axis=1, keepdims=True) * 0.2        # sphere shell
    organ = rng.normal(size=(400, 3)) * 0.04 + np.array([0.0, 0.0, 0.05])  # interior blob
    ids, w = bind_structure(organ, skin, cKDTree(skin))
    R = Rot.from_rotvec([0.0, 0.4, 0.1]).as_matrix(); t = np.array([0.1, 0.0, -0.2])
    posed = skin @ R.T + t
    Rk, cs, cd = weighted_kabsch(skin[ids.astype(int)], posed[ids.astype(int)], w.astype(float))
    moved = (organ - cs) @ Rk.T + cd
    assert np.abs(moved - (organ @ R.T + t)).max() < 0.02          # < 2 cm for a 0.4 rad rotation


def test_group_table_covers_ts_names():
    for n, g in [("vertebrae_L3", "spine"), ("rib_left_7", "ribcage"), ("aorta", "artery"),
                 ("portal_vein_and_splenic_vein", "vein"), ("lung_lower_lobe_left", "lung"),
                 ("gluteus_maximus_left", "muscle"), ("femur_right", "pelvis_leg"), ("liver", "gi"),
                 ("kidney_left", "urinary"), ("skull", "skull"), ("heart_myocardium", "heart")]:
        assert group_of(n) == g, (n, group_of(n))


ATLAS = os.environ.get("ATLAS_PREFIX")
needs_atlas = pytest.mark.skipif(not ATLAS, reason="set ATLAS_PREFIX to a built atlas to run")


@needs_atlas
def test_built_atlas_is_total_segmentator_scale_and_inside_body():
    from anatomy.runtime import Atlas
    a = Atlas(ATLAS)
    names = a.names()
    assert len(names) >= 100                                  # TotalSegmentator-scale coverage
    assert {"liver", "aorta", "spleen", "stomach", "heart_myocardium"} & set(names)
    pos = np.vstack([s["pos"] for s in a.structures])
    assert 1.2 < pos[:, 1].min() * -1 < 1.8                   # head within MHR height (y down)
    assert max(len(s["faces"]) for s in a.structures) < 40000


@needs_atlas
def test_atlas_posing_identity_and_translation():
    from anatomy.runtime import Atlas
    import json
    a = Atlas(ATLAS)
    rest = np.zeros((a.meta["n_skin_vertices"], 3))
    for s in a.structures:                                    # fill support vertices at rest
        rest[s["s_idx"]] = s["s_rest"]
    ident = a.pose(rest)
    for s in a.structures[:20]:
        assert np.abs(ident[s["name"]] - s["pos"]).max() < 1e-4
    shifted = a.pose(rest + np.array([0.0, 0.0, -0.5]))
    s0 = a.structures[0]
    assert np.allclose(shifted[s0["name"]] - s0["pos"], [0, 0, -0.5], atol=1e-4)


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
