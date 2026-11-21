"""How well does each stage of the depth fit (depth/fit.py) bring the mesh onto the measured surface, on recordings?

For every frame the fit sees a random 70% of the vertices; the error is measured on the other 30%, which the fit never
compared with the depth (they move only because their neighbours and their body part do). Error = distance along the camera
ray between the vertex and the depth surface it projects onto, in mm.

usage: python tools/evaluate_fit.py mhr_model.pt ../recordings/NAME.npz [...] [--every N]
"""
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_depth_recording import load  # noqa: E402
from depth.fit import BodyFit, DepthFrame  # noqa: E402

# Depth lens model of the Femto Bolt these recordings were made with (narrow unbinned mode). Recordings made before the
# calibration carried it ("kd", "dist") use these.
BOLT_KD = (505.34368896484375, 505.4107666015625, 336.45892333984375, 325.3226623535156)
BOLT_DIST = (15.141837120056152, 8.009384155273438, 1.3242264685686678e-05, -2.4030552594922483e-05,
             0.3177873492240906, 15.448055267333984, 13.116472244262695, 1.9120935201644897)


def frame_of(z, cal, i):
    return DepthFrame(z["depth"][i], z["rays"], cal["R"], cal["t"], cal.get("kd", BOLT_KD), cal.get("dist", BOLT_DIST))


def stats(e):
    e = np.asarray(e)
    return (float(np.sqrt(np.mean(e ** 2))), float(np.median(np.abs(e))), float(np.percentile(np.abs(e), 90))) if len(e) else (np.nan,) * 3


def main():
    argv, args = sys.argv[1:], []
    while argv:
        a = argv.pop(0)
        if a.startswith("--"):
            argv.pop(0)                                          # every option here takes one value
        else:
            args.append(a)
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    from bind_mhr_joints import MHR
    mhr = MHR(args[0])
    names = list(mhr.m.character_torch.skeleton.joint_names)
    rng = np.random.default_rng(0)
    for path in args[1:]:
        z, cal = load(path)
        fit = BodyFit(z["faces"], names, mhr.w_joint, mhr.w_vert, mhr.w)
        rows, ms = {s: [] for s in ("model", "one scale", "1 distance", "2 pose", "3 surface", "3 surface, tracking")}, []
        ms_track, prevM = [], None
        within = {2: [], 5: []}
        for i in range(0, len(z["verts"]), every):
            if not z["has_depth"][i]:
                continue
            V = (z["verts"][i].astype(np.float64) + z["cam_t"][i].astype(np.float64)) * 1000.0
            depth = frame_of(z, cal, i)
            use = rng.random(len(V)) < 0.7
            held = ~use
            s, a = fit.part_scales(V, depth, use)
            if s is None:
                continue
            g = a["global_scale"]
            t0 = time.perf_counter()
            X3, M3, info = fit.fit(V, depth, use)
            ms.append((time.perf_counter() - t0) * 1000)
            X1 = V * (fit.W @ s)[:, None]
            # tracking: this frame's fit starts from the previous frame's transforms and takes one pose step
            if prevM is not None:
                t1 = time.perf_counter()
                Xt, Mt, _ = fit.fit(V, depth, use, init=prevM, pose_iters=1)
                ms_track.append((time.perf_counter() - t1) * 1000)
            else:
                Xt, Mt = X3, M3
            prevM = M3 if Mt is None else Mt
            X2, _, _ = fit.fit(V, depth, use, surface=False)
            e3 = np.abs(fit.residual(X3, depth, held, 80.0))
            for lim_mm in within:
                within[lim_mm].append(100.0 * (e3 < lim_mm).mean())
            for name, X, lim in (("model", V, 400.0), ("one scale", V * g, 80.0), ("1 distance", X1, 80.0), ("2 pose", X2, 80.0), ("3 surface", X3, 80.0), ("3 surface, tracking", Xt, 80.0)):
                rows[name].append(stats(fit.residual(X, depth, held, lim)) + (len(fit.residual(X, depth, held, lim)),))
        print(f"{os.path.basename(path)}: {len(ms)} frames, fit from scratch {np.median(ms):.0f} ms, tracking fit {np.median(ms_track):.0f} ms (median)")
        print(f"   {'held-out vertices: error along the ray, mm':44s} {'rms':>7s} {'median':>7s} {'90th':>7s}   n")
        for name, r in rows.items():
            r = np.array(r)
            print(f"   {name:44s} {np.median(r[:, 0]):7.1f} {np.median(r[:, 1]):7.1f} {np.median(r[:, 2]):7.1f}   {np.median(r[:, 3]):.0f}")
        print(f"   3 surface: {np.median(within[2]):.0f}% of held-out vertices within 2 mm, {np.median(within[5]):.0f}% within 5 mm")


if __name__ == "__main__":
    main()
