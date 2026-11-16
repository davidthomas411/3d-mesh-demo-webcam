"""Side (lateral) views of the best model through a motion: lateral bone radiograph and midline sagittal reformat (real HU), with the posed
mesh's vertebra-to-vertebra gaps printed for the neck. usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/render_side.py OUT.png [phase] [u,u,u]"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv


def side_panels(version, posed, skin, centre, x_mid, PW=int(os.environ.get("PW", 300))):
    import render_radiology as rr
    W, CFG = bv.W, bv.CFG
    c, (y0, y1) = CFG["c"], CFG["y"]
    fld = W.field(posed, skin, structures=(version != "V0"))
    region = (np.array([c[0] - 260, y0, c[2] - 560.0]), np.array([c[0] + 260, y1, c[2] + 220.0]))
    vol, bone, axes, st = W.warped_volume(fld, step=float(os.environ.get("CTW_STEP", 2.0)), region=region, soft_bone=True)
    L = rr.radiograph(vol, bone, 0, st)                                   # sum over x -> (y, z); row 0 = head, column 0 = front (-z)
    u8 = rr.unsharp(rr.window(L, 0.0, 20.0, 0.75), 2.2, 1.0)
    lat = Image.fromarray(u8).convert("RGB").resize((PW, int(u8.shape[0] * PW / u8.shape[1])), Image.BICUBIC)
    cs = centre.copy(); cs[2] -= 100.0                                      # shifted forward: a bent-forward head and chest leave the standing frame
    P, shape = W.plane(0, x_mid, 2, 1, cs, 280.0, (y1 - y0) / 2, 2.0)  # sagittal plane through the posed T9: horizontal = z, vertical = y
    src = W.source_positions(fld, P)
    hu = W.sample(W.hu_c, src).reshape(shape).astype(np.float32)
    g = rr.unsharp(rr.window(hu, 40 - 175, 40 + 175), 1.4, 0.6)
    sag = Image.fromarray(np.stack([g] * 3, -1)).resize((PW, int(g.shape[0] * PW / g.shape[1])), Image.LANCZOS)
    return lat, sag


if __name__ == "__main__":
    out = sys.argv[1]
    phase = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    us = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.0, 0.25, 0.5, 0.75]
    bv.setup()
    W, CFG = bv.W, bv.CFG
    atlas = W.atlas
    by = {s["name"]: s["pos"] * 1000 for s in atlas.structures}
    cols = []
    for u in us:
        name, p = bv.motion_params(phase, u)
        skin = bv.SESS.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
        posed = bv.posed_for("V4", atlas, skin, bv.PS)
        s_, R_, t_ = W.T["vertebrae_T9"]
        c9 = (((posed["vertebrae_T9"] - t_) @ R_) / s_).mean(0)
        centre = np.array([CFG["c"][0], (CFG["y"][0] + CFG["y"][1]) / 2, CFG["c"][2]])
        lat, sag = side_panels("V4", posed, skin, centre, c9[0])
        P = {n: v * 1000 for n, v in posed.items()}
        gap = lambda a, b: float(cKDTree(P[a]).query(P[b])[0].min())
        gaps = {"C7-T1": gap("vertebrae_C7", "vertebrae_T1"), "C6-C7": gap("vertebrae_C6", "vertebrae_C7"), "C1-skull": gap("skull", "vertebrae_C1")}
        print(f"u={u:.2f} {name}: neck gaps (mm) " + ", ".join(f"{k} {v:.1f}" for k, v in gaps.items()), flush=True)
        cols.append((u, gaps, lat, sag))
    PW = int(os.environ.get("PW", 300))
    hh = max(c[2].height + c[3].height for c in cols)
    sheet = Image.new("RGB", (len(cols) * (PW + 6), hh + 70), (0, 0, 0)); d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"Best model (V4, atlas v8), {name}: lateral bone radiograph (top) and midline sagittal reformat, real HU (bottom). Face to the left.", fill=(235, 235, 235))
    for i, (u, gaps, lat, sag) in enumerate(cols):
        x = i * (PW + 6)
        d.text((x + 6, 28), f"u = {u:.2f}   C7-T1 gap {gaps['C7-T1']:.1f} mm   C1-skull {gaps['C1-skull']:.1f} mm", fill=(255, 220, 120))
        sheet.paste(lat, (x, 60)); sheet.paste(sag, (x, 60 + lat.height))
    sheet.save(out); print("wrote", out, sheet.size)
