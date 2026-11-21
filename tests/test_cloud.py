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
