"""Point-cloud helpers of the Femto Bolt stream (depth/cloud.py): no camera needed."""
import numpy as np
from depth.cloud import subsample, to_colour_frame, pack, FemtoCloud


def test_subsample_keeps_valid_range_and_caps_the_count():
    rng = np.random.default_rng(0)
    p = rng.uniform(-1000, 1000, (5000, 3)); p[:, 2] = rng.uniform(0, 6000, 5000)
    p[:100, 2] = 0                                              # invalid depth
    out = subsample(p, 800)
    assert len(out) == 800 and out[:, 2].min() > 250 and out[:, 2].max() < 4500
    few = subsample(p[:300], 800)
    assert 0 < len(few) <= 300 and (few[:, 2] > 250).all()


def test_points_go_to_the_colour_frame_and_survive_packing_to_the_millimetre():
    th = np.radians(6.0)                                        # the Bolt's depth camera is tilted about 6 degrees
    R = np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]]); t = np.array([-32.0, -1.0, 2.0])
    p = np.array([[0.0, 0.0, 1000.0], [100.0, -200.0, 1500.0]])
    q = to_colour_frame(p, R, t)
    assert np.allclose(q[0], [-32.0, 1000 * np.sin(th) - 1.0, 1000 * np.cos(th) + 2.0])
    back = np.frombuffer(pack(q), "<i2").reshape(-1, 3)
    assert np.abs(back - q).max() <= 0.5


def test_a_missing_camera_is_reported_not_raised(monkeypatch):
    import builtins, time
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError("no sdk")) if n == "pyorbbecsdk" else real(n, *a, **k))
    c = FemtoCloud()
    assert c.get() is None
    for _ in range(50):
        if c.error: break
        time.sleep(0.05)
    assert c.error and "ImportError" in c.error and c.get() is None


def test_live_anchor_recovers_the_scale_of_a_body_from_a_depth_map_made_of_points():
    """The mesh is 8% too small and too near; the depth camera sees the true body as a point cloud."""
    import os, time
    import pytest
    from depth.live import LiveAnchor, depth_map_from_points, fast_visible
    rest = os.path.join(os.path.dirname(__file__), "..", "..", "Fast-SAM-3D-Body", "mhr_rest.npz")
    if not os.path.exists(rest):
        pytest.skip("no neutral body mesh on this machine")
    d = np.load(rest); V, F = d["verts"].astype(np.float64), d["faces"]
    P = (V + np.array([0.05, 0.9, 2.4])) * 1000.0               # camera frame, mm: y down, the body 2.4 m away
    hw, f = (180, 320), 753.14 * 320 / 1280
    K = np.array([[f, 0, 160.0], [0, f, 97.0], [0, 0, 1.0]])
    vis = fast_visible(P, F, K, hw)
    assert 0.25 < vis.mean() < 0.6 and (P[vis][:, 2] < np.median(P[:, 2]) + 80).mean() > 0.8     # the front of the body
    true = P * 1.08
    rng = np.random.default_rng(0)
    front = true[fast_visible(true, F, K, hw)]
    cloud = np.repeat(front, 6, 0) + rng.normal(0, 2.0, (len(front) * 6, 3))                     # what the depth camera measures
    depth = depth_map_from_points(cloud, K, hw)
    a = LiveAnchor(F)
    t = time.perf_counter(); info = a.update(P, depth, K); ms = (time.perf_counter() - t) * 1000
    assert info["ok"] and abs(info["raw"] - 1.08) < 0.01 and info["rms_mm"] < 25, info
    assert ms < 150, ms                                         # it runs inside the inference loop
    for _ in range(20):
        info = a.update(P, depth, K)
    assert abs(info["scale"] - 1.08) < 0.01
    wall = a.update(P, np.full(hw, 3100.0) + rng.normal(0, 150, hw), K)      # a depth image that is not this body: refused
    assert not wall["ok"] and "does not match" in wall["reason"] and wall["scale"] == info["scale"]
    empty = a.update(P, np.zeros(hw), K)                        # nobody in the depth image: keep the last scale, say why
    assert not empty["ok"] and empty["scale"] == info["scale"] and empty["reason"]
