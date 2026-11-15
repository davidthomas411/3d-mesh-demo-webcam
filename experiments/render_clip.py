"""Drive the skeleton and the real CT with a motion-capture clip of MHR parameters (see tools/mhr_clip.py).
Per frame: AP bone radiograph | lateral bone radiograph | coronal reformat | sagittal reformat (real HU), best model (V4).
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/render_clip.py CLIP.npz OUTDIR [--workers 2] [--only i,j,k]"""
import os, sys
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
import render_side as rs
from mhr_clip import load_clip, prepare

PARAMS = LABEL = None
OUT = "."


def job(i):
    import render_radiology as rr
    W, CFG = bv.W, bv.CFG
    p = PARAMS[i][None].astype(np.float32)
    skin = bv.SESS.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = bv.posed_for("V4", W.atlas, skin, bv.PS)
    s_, R_, t_ = W.T["vertebrae_T9"]
    c9 = (((posed["vertebrae_T9"] - t_) @ R_) / s_).mean(0)
    centre = np.array([CFG["c"][0], (CFG["y"][0] + CFG["y"][1]) / 2, CFG["c"][2]])
    ap, cor = bv.panel("V4", posed, skin, c9[2] - 55.0, centre)
    lat, sag = rs.side_panels("V4", posed, skin, centre, c9[0], PW=bv.PW)
    cols = [(ap, "AP bone radiograph"), (lat, "lateral bone radiograph"), (cor, "coronal, real HU"), (sag, "sagittal, real HU")]
    top, gap = 64, 6
    H = max(a.height for a, _ in cols)
    cv = Image.new("RGB", (len(cols) * (bv.PW + gap), H + top + 22), (0, 0, 0)); d = ImageDraw.Draw(cv)
    d.text((8, 6), f"{LABEL}   frame {i + 1}/{len(PARAMS)}", fill=(255, 220, 120))
    d.text((8, 24), "Skeleton (V4, atlas v8) and real CT driven by per-frame MHR parameters. Synthetic image data, not for clinical use.", fill=(170, 200, 230))
    for k, (im, name) in enumerate(cols):
        x = k * (bv.PW + gap)
        d.text((x + 4, 44), name, fill=(235, 235, 235)); cv.paste(im, (x, top))
    cv.save(os.path.join(OUT, f"{i:03d}.png"))
    return i


if __name__ == "__main__":
    a = sys.argv[1:]
    arr, fps, source = load_clip(a[0])
    PARAMS = prepare(arr); LABEL = source[:90]
    OUT = a[1]; os.makedirs(OUT, exist_ok=True)
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
    bv.PW = int(os.environ.get("PW", 300))
    idx = [int(x) for x in a[a.index("--only") + 1].split(",")] if "--only" in a else list(range(len(PARAMS)))
    bv.setup()
    print(f"clip: {len(arr)} frames at {fps:g} fps, source: {source}", flush=True)
    with Pool(workers) as pool:
        for i in pool.imap_unordered(job, idx):
            print("frame", i, flush=True)
    print("finished", flush=True)
