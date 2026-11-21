"""Figure of the depth fit on a recording: the measured surface, the model's mesh and the fitted mesh, from above and from the
side (no picture of the person), and the held-out error of each stage.

usage: python tools/fit_figure.py mhr_model.pt ../recordings/NAME.npz OUT.png [frame]
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_depth_recording import load  # noqa: E402
from evaluate_fit import frame_of  # noqa: E402
from depth.cloud import points_from_depth, to_colour_frame  # noqa: E402
from depth.fit import BodyFit  # noqa: E402


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bind_mhr_joints import MHR
    mhr = MHR(sys.argv[1])
    z, cal = load(sys.argv[2])
    i = int(sys.argv[4]) if len(sys.argv) > 4 else len(z["verts"]) // 2
    fit = BodyFit(z["faces"], list(mhr.m.character_torch.skeleton.joint_names), mhr.w_joint, mhr.w_vert, mhr.w)
    V = (z["verts"][i].astype(np.float64) + z["cam_t"][i].astype(np.float64)) * 1000.0
    fr = frame_of(z, cal, i)
    rng = np.random.default_rng(0)
    use = rng.random(len(V)) < 0.7
    s, a = fit.part_scales(V, fr, use)
    X1 = V * (fit.W @ s)[:, None]
    X2, _, _ = fit.fit(V, fr, use, surface=False)
    X3, _, _ = fit.fit(V, fr, use)
    pts = to_colour_frame(points_from_depth(z["depth"][i], z["rays"]), np.array(cal["R"]).reshape(3, 3), np.array(cal["t"]))
    seen = fit.observe(X3, fr)["ok"]
    cy, cx = np.median(X3[seen][:, 1]), np.median(X3[seen][:, 0])
    zlo, zhi = X3[seen][:, 2].min() - 150, X3[seen][:, 2].max() + 350
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5))
    for a_, (h, lab, band_axis, centre) in zip(ax[:2], ((0, "from above: x (mm)", 1, cy), (1, "from the side: y, down (mm)", 0, cx))):
        band = np.abs(pts[:, band_axis] - centre) < 25
        a_.plot(pts[band][:, h], pts[band][:, 2], ".", ms=1.2, color="0.55", label="depth camera (a 50 mm slab)")
        for X, col, name in ((V, "tab:red", "mesh from the picture alone"), (V * a["global_scale"], "tab:orange", "one scale"), (X3, "tab:green", "fitted")):
            m = np.abs(X[:, band_axis] - (centre if X is not V else centre / a["global_scale"])) < 25
            a_.plot(X[m][:, h], X[m][:, 2], ".", ms=1.6, color=col, label=name)
        a_.set_ylim(zhi, zlo); a_.set_xlabel(lab); a_.set_ylabel("distance from the camera (mm)"); a_.set_aspect("equal")
        lim = (cx - 330, cx + 330) if h == 0 else (cy - 330, cy + 330)
        a_.set_xlim(*lim); a_.legend(markerscale=6, loc="lower right", fontsize=8)
    held = ~use
    bins = np.linspace(-30, 30, 121)
    for X, col, name in ((V * a["global_scale"], "tab:orange", "one scale"), (X1, "tab:purple", "1 distance per part"), (X2, "tab:blue", "2 + articulated pose"), (X3, "tab:green", "3 + surface")):
        e = fit.residual(X, fr, held)
        ax[2].hist(e, bins=bins, histtype="step", color=col, lw=1.6, label=f"{name}: median |e| {np.median(np.abs(e)):.1f} mm, 90% within {np.percentile(np.abs(e), 90):.1f} mm")
    ax[2].set_xlabel("held-out vertices: measured surface minus mesh, along the ray (mm)"); ax[2].set_ylabel("vertices")
    ax[2].legend(fontsize=8, loc="upper left"); ax[2].set_title(f"frame {i} of {os.path.basename(sys.argv[2])}")
    plt.tight_layout(); plt.savefig(sys.argv[3], dpi=80)
    print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
