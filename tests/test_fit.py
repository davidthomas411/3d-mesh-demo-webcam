"""Depth fit of the body mesh (depth/fit.py) on a synthetic scene: a known body in front of a known depth camera.

Needs the body model on this machine (mhr_model.pt and the neutral mesh); skipped without them."""
import os

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
FAST = os.path.join(HERE, "..", "..", "Fast-SAM-3D-Body")
MHR_PATH = os.path.join(FAST, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")
REST = os.path.join(FAST, "mhr_rest.npz")
pytestmark = pytest.mark.skipif(not (os.path.exists(MHR_PATH) and os.path.exists(REST)), reason="no body model on this machine")

KD = (505.0, 505.0, 320.0, 288.0)
W, H = 640, 576


@pytest.fixture(scope="module")
def body():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    from bind_mhr_joints import MHR
    from depth.fit import BodyFit
    mhr = MHR(MHR_PATH)
    d = np.load(REST)
    fit = BodyFit(d["faces"], list(mhr.m.character_torch.skeleton.joint_names), mhr.w_joint, mhr.w_vert, mhr.w)
    V = (d["verts"].astype(np.float64) + np.array([0.02, 1.35, 1.1])) * 1000.0      # the chest in front of the camera, 1.1 m away
    return fit, V


def depth_of(fit, X, noise_mm=0.0, seed=0):
    """What an ideal pinhole depth camera at the colour camera's place sees of the mesh X: a DepthFrame."""
    from depth.fit import DepthFrame
    from depth.synth import rasterize_depth
    K = np.array([[KD[0], 0, KD[2]], [0, KD[1], KD[3]], [0, 0, 1.0]])
    z = rasterize_depth(X, fit.faces, K, (H, W))
    z[~np.isfinite(z)] = 0.0
    if noise_mm:
        z = z + (z > 0) * np.random.default_rng(seed).normal(0, noise_mm, z.shape)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - KD[2]) / KD[0], (v - KD[3]) / KD[1]], -1).astype(np.float32)
    return DepthFrame(np.round(z).astype(np.uint16), rays, np.eye(3), np.zeros(3), KD, np.zeros(8))


def test_joints_are_assigned_to_parts_by_their_longest_prefix():
    from depth.fit import PARTS, joint_parts
    names = [p[0] for p in PARTS]
    jp = joint_parts(["l_wrist", "l_wrist_twist", "l_lowarm_twist2_proc", "c_spine3", "r_thumb2", "c_head_null", "root"])
    assert [names[g] for g in jp] == ["hand_l", "lowarm_l", "lowarm_l", "chest", "hand_r", "head", "pelvis"]


def _leaning_truth(fit, V, lean_deg=6.0, head_deg=5.0, scale=0.88):
    """The true body: nearer than the model thought, leaning, and with the head turned a little more than the model's."""
    from scipy.spatial.transform import Rotation as Rot
    c = V[fit.W[:, 2] > 0.5].mean(0)                             # about the chest
    truth = (V - c) @ Rot.from_euler("x", lean_deg, degrees=True).as_matrix().T + c
    head = fit.W[:, 4:5]
    ch = V[fit.W[:, 4] > 0.5].mean(0)
    truth = (1 - head) * truth + head * ((truth - ch) @ Rot.from_euler("y", head_deg, degrees=True).as_matrix().T + ch)
    return truth * scale


def test_the_body_is_placed_as_one_and_its_skin_drawn_onto_the_surface(body):
    """The default fit: one transform for the whole body. The model's pose is kept (a head the truth has turned further is
    NOT turned by the depth), and the skin still ends on the measured surface."""
    fit, V = body
    truth = _leaning_truth(fit, V)
    frame = depth_of(fit, truth)
    before = fit.residual(V, frame, max_mm=400.0)
    track = {}
    X, M, info = fit.fit(V, frame, pose_iters=4, track=track)
    after = fit.residual(X, frame)
    assert info["ok"] and M.shape == (fit.G, 4, 4)
    assert np.abs(M - M[0]).max() < 1e-12                        # every part has the same transform: nothing turns on its own
    assert abs(info["part_scale"][1] - 0.88) < 0.03
    assert np.median(np.abs(before)) > 80 and np.median(np.abs(after)) < 1.0 and np.percentile(np.abs(after), 90) < 5.0
    # the placed mesh (what the skeleton is built on) is the model's mesh moved rigidly and scaled: its shape is untouched
    P = track["X_pose"]
    d0 = np.linalg.norm(V[::97, None] - V[None, ::97], axis=-1); d1 = np.linalg.norm(P[::97, None] - P[None, ::97], axis=-1)
    k = np.cbrt(np.linalg.det(M[0, :3, :3]))
    assert np.abs(d1 - k * d0).max() < 1e-6
    # and the whole-body lean is followed in part only: the depth turns the body no further than it insists
    from scipy.spatial.transform import Rotation as Rot
    turned = np.degrees(Rot.from_matrix(M[0, :3, :3] / k).magnitude())
    assert 1.0 < turned < 6.5


def test_per_part_fitting_is_still_available_and_fits_closer(body):
    fit, V = body
    truth = _leaning_truth(fit, V)
    frame = depth_of(fit, truth)
    X, M, info = fit.fit(V, frame, pose_iters=4, articulated=True)
    seen = fit.observe(X, frame)["ok"]
    assert np.median(np.abs(fit.residual(X, frame))) < 1.0
    assert np.median(np.linalg.norm(X - truth, axis=1)[seen]) < 6.0      # near the true vertices in space, not only along the rays
    assert np.abs(M - M[0]).max() > 1e-3                         # the parts did move separately


def test_tracking_follows_a_small_motion_in_one_step_and_then_stands_still(body):
    fit, V = body
    truth = V * 0.9
    X, M, info = fit.fit(V, depth_of(fit, truth), pose_iters=4)
    shifted = truth + np.array([6.0, 0.0, 4.0])                  # the person moves 6 mm sideways and 4 mm back
    frame = depth_of(fit, shifted)
    for _ in range(4):
        X, M, info = fit.fit(V, frame, init=M, pose_iters=1)
    seen = fit.observe(X, frame)["ok"]
    assert np.median(np.abs(fit.residual(X, frame))) < 1.0
    assert abs(np.median((X - shifted)[seen][:, 0])) < 2.5       # the sideways motion was followed, not just the depth
    X2, M2, info2 = fit.fit(V, frame, init=M, pose_iters=1)      # nothing changes: the fit does not creep
    for _ in range(6):
        X2, M2, info2 = fit.fit(V, frame, init=M2, pose_iters=1)
    X3, M3, info3 = fit.fit(V, frame, init=M2, pose_iters=1)
    assert info3["moved_mm"] == 0.0 and info3["reused"] and np.abs(X3 - X2).max() < 1e-6


def test_joints_move_with_their_parts(body):
    fit, V = body
    M = np.tile(np.eye(4), (fit.G, 1, 1))
    M[:, :3, :3] *= 0.9
    M[4, :3, 3] = [10.0, 0.0, 0.0]                               # the head part also shifts 10 mm
    nj = len(fit.jpart)
    pos = np.random.default_rng(0).normal(0, 300, (nj, 3)) + [0, 0, 1500]
    rot = np.tile(np.eye(3), (nj, 1, 1))
    P, R = fit.move_joints(pos, rot, M)
    head = fit.jpart == 4
    assert np.allclose(P[~head], 0.9 * pos[~head]) and np.allclose(P[head], 0.9 * pos[head] + [10.0, 0, 0])
    assert np.allclose(R, rot)                                   # a scale does not turn a joint


def test_nobody_in_the_depth_image_is_reported(body):
    fit, V = body
    from depth.fit import DepthFrame
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - KD[2]) / KD[0], (v - KD[3]) / KD[1]], -1).astype(np.float32)
    X, M, info = fit.fit(V, DepthFrame(np.zeros((H, W), np.uint16), rays, np.eye(3), np.zeros(3), KD, np.zeros(8)))
    assert M is None and not info["ok"] and "not in the depth image" in info["reason"] and X is V
