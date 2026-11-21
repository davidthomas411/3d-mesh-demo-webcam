"""Picture of a tracking run (gantry_track.py): the shifts through the live pass, and slices through the head showing the
live frame's points over the reference frame's from the same gantry angle.

    python tools/gantry_track_plot.py FOLDER OUT_DIR [--json track_linac.json] [--slices 5010,5226,5478]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gantry_recording import GantryRecording  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gantry_cube import CubeGantry, frame_angles
    from gantry_geometry import GantryGeometry
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    rec = GantryRecording(sys.argv[1]); out = sys.argv[2]
    d = json.load(open(os.path.join(out, arg("--json", "track_linac.json"))))
    rows = d["frames"]; fr = np.array([r["frame"] for r in rows]); t = (fr - fr[0]) / 30.0
    ang = np.array([r["gantry_deg"] for r in rows])
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.38, wspace=0.22)
    cols = {"x (left-right)": "#c0392b", "y (along couch)": "#2471a3", "z (up)": "#1e8449"}
    for c, region in enumerate(("head", "chest")):
        for r_, ref in enumerate(("matched", "fused")):
            ax = fig.add_subplot(gs[r_, c])
            S = np.array([r[region][ref]["shift_mm"] for r in rows])
            for k, (name, col) in enumerate(cols.items()):
                ax.plot(t, S[:, k], "-o", ms=3, color=col, label=name)
            ax.plot(t, np.linalg.norm(S, axis=1), "k--", lw=1, label="size")
            ax.axhline(0, color="#999", lw=0.6); ax.set_ylim(-4, 8); ax.grid(alpha=0.25)
            ax.set_title("%s, %s reference" % (region, ref), fontsize=11)
            ax.set_ylabel("apparent shift, mm")
            if r_ == 1:
                ax.set_xlabel("seconds into the return pass")
            if c == 0 and r_ == 0:
                ax.legend(fontsize=8, ncol=2, loc="upper left")
            top = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, t, ang), lambda a: np.interp(a, ang[::-1], t[::-1])))
            top.set_xlabel("gantry angle, degrees", fontsize=8); top.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[0:2, 2]); ax.axis("off")
    sm = d["summary"]
    lines = ["Outward pass = reference, return pass = live.", "Each live frame measured on its own.", "Gantry angle from: %s" % sm.get("gantry_angle_from", "?"), ""]
    for k, v in sm.items():
        if isinstance(v, dict):
            lines += [k, "   median %.1f mm, 95%% %.1f mm, turn %.2f deg" % (v["apparent_shift_mm_median"], v["p95"], v["apparent_turn_deg_median"])]
    lines += ["", "Nothing independent measured the person,", "so these are upper bounds on the error.", "",
              "Below: a 30 mm slice across the head,", "seen from the feet. Grey = reference frame,", "red = live frame, same gantry angle."]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
    # slices through the head
    geo = GantryGeometry(os.path.join(out, "geometry.npz")); cg = CubeGantry(os.path.join(out, "cube_geometry.npz"))
    fa, _ = frame_angles(out)
    refs = list(range(4500, 5004, 3)); rth = np.array([fa(i) for i in refs])

    def sl(i):
        P, _ = rec.points(rec.depth(i), mask=~geo.fixed, step=1)
        X = cg.to_room(P, fa(i))
        m = (X[:, 1] > -365) & (X[:, 1] < -335) & (np.abs(X[:, 0]) < 130) & (X[:, 2] > -30) & (X[:, 2] < 180)
        return X[m]
    picks = [int(x) for x in arg("--slices", "5010,5226,5478").split(",")]
    for c, i in enumerate(picks):
        ax = fig.add_subplot(gs[2, c])
        j = refs[int(np.argmin(np.abs(rth - fa(i))))]
        A, B = sl(j), sl(i)
        ax.plot(A[:, 0], A[:, 2], ".", ms=2.5, color="#555", label="reference (frame %d)" % j)
        ax.plot(B[:, 0], B[:, 2], ".", ms=2.5, color="#e74c3c", alpha=0.7, label="live (frame %d)" % i)
        row = min(rows, key=lambda r: abs(r["frame"] - i))
        s = row["head"]["matched"]["shift_mm"]
        ax.set_title("gantry %.0f deg, %.0f s later: measured x %+.1f, z %+.1f mm" % (fa(i), (i - j) / 30.0, s[0], s[2]), fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("room x, mm"); ax.set_ylabel("room z, mm")
        ax.set_xticks(np.arange(-120, 121, 20)); ax.tick_params(labelsize=7); ax.legend(fontsize=8, loc="lower center")
    fig.suptitle("Tracking while the gantry turns: return pass against the outward pass", fontsize=14, y=0.97)
    p = os.path.join(out, "track.png"); fig.savefig(p, dpi=110, bbox_inches="tight"); print(p)


if __name__ == "__main__":
    main()
