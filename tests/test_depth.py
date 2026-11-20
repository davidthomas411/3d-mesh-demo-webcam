import numpy as np
from depth.synth import make_K, phantom_torso, render_depth, vertex_normals
from depth.anchor import visible_vertices, depth_anchor
from depth.icp import projective_icp, roi_icp_cloud
from depth.preprocess import reject_flying, remove_support_plane

HW = (240, 320)
K = make_K(HW[1], HW[0], 70)


def place(v, z=1800.0, rot=(0, 0, 0), t=(0, 0, 0)):
    from scipy.spatial.transform import Rotation as R
    Rm = R.from_euler("xyz", rot, degrees=True).as_matrix()
    return (v @ Rm.T) + np.array([t[0], t[1], z + t[2]])


def test_anchor_recovers_scale():
    v, f = phantom_torso(n_lat=20, n_lon=40)
    truth = place(v, 1800)
    depth = render_depth(truth, f, K, HW, noise_mm=3)
    wrong = truth * 0.8          # monocular ambiguity: right overlay, wrong metric depth
    vis = visible_vertices(wrong, f, K, HW)
    out, info = depth_anchor(wrong, depth, K, vis)
    assert info["ok"] and abs(info["value"] - 1.25) < 0.01
    assert np.abs(out - truth).mean() < 8.0


def test_anchor_translate_mode():
    v, f = phantom_torso(n_lat=20, n_lon=40)
    truth = place(v, 1800)
    depth = render_depth(truth, f, K, HW)
    off = truth + np.array([0, 0, -150.0])
    out, info = depth_anchor(off, depth, K, visible_vertices(off, f, K, HW), mode="translate")
    assert abs(info["value"] - 150) < 20          # perspective makes this approximate


def test_icp_recovers_6dof():
    v, f = phantom_torso(n_lat=20, n_lon=40)
    ref = place(v, 1800)
    live_true = place(v, 1800, rot=(3, -4, 2), t=(8, -5, 12))
    depth = render_depth(live_true, f, K, HW, noise_mm=2)
    T, info = projective_icp(ref, vertex_normals(ref, f), depth, K, iters=15)
    est = ref @ T[:3, :3].T + T[:3, 3]
    vis = visible_vertices(live_true, f, K, HW)
    err = np.linalg.norm(est - live_true, axis=1)[vis]
    assert info["ok"] and err.mean() < 4.0, err.mean()


def test_icp_with_couch_and_flying_pixels():
    v, f = phantom_torso(n_lat=20, n_lon=40)
    ref = place(v, 1800)
    live_true = place(v, 1800, rot=(2, 3, -2), t=(-6, 4, 9))
    depth = render_depth(live_true, f, K, HW, noise_mm=2, flying_px=0.4)
    depth = np.where(depth == 0, 2150, depth).astype(np.uint16)       # couch plane behind
    depth, plane = remove_support_plane(depth, K, tol_mm=10)
    depth = reject_flying(depth, 60)
    T, info = projective_icp(ref, vertex_normals(ref, f), depth, K, iters=15)
    est = ref @ T[:3, :3].T + T[:3, 3]
    vis = visible_vertices(live_true, f, K, HW)
    err = np.linalg.norm(est - live_true, axis=1)[vis].mean()
    assert err < 8.0, err


def test_cloth_compensation():
    v, f = phantom_torso(n_lat=20, n_lon=40)
    ref = place(v, 1800)
    gown = ref + vertex_normals(ref, f) * 12            # cloth 12 mm proud of skin
    depth = render_depth(gown, f, K, HW, noise_mm=1)
    N = vertex_normals(ref, f)
    shift = lambda T: np.linalg.norm(((ref @ T[:3, :3].T + T[:3, 3]) - ref).mean(0))
    T0, _ = projective_icp(ref, N, depth, K, iters=10)
    T1, _ = projective_icp(ref, N, depth, K, iters=10, cloth_mm=12.0)
    assert shift(T0) > 6.0 and shift(T1) < 4.0, (shift(T0), shift(T1))


def test_fusion_end_to_end():
    from depth.fusion import DepthFusion, mhr_to_camera_mm
    v, f = phantom_torso(n_lat=20, n_lon=40)
    truth = place(v, 1800, rot=(1, 2, -1), t=(5, -4, 6))
    depth = render_depth(truth, f, K, HW, noise_mm=2, dropout=0.05)
    mono = truth * 0.85 + np.array([0, 0, 0.0])        # SAM-like: right pixels, wrong metric depth
    fu = DepthFusion(f, K, icp_iters=4)
    for _ in range(5):                                  # EMA settles
        out, info = fu.process(mono, depth)
    vis = visible_vertices(truth, f, K, HW)
    assert info["ok"] and np.linalg.norm(out - truth, axis=1)[vis].mean() < 8.0
    # coordinate helper round-trips the documented convention
    m = mhr_to_camera_mm(np.array([[0.0, 0.1, 0.0]]), np.array([0.0, 0.0, -2.0]))
    assert np.allclose(m, [[0, -100.0, 2000.0]])
