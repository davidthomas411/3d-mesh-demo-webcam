"""Same rig, an MRI instead of a CT: drive the TotalSegmentator-MRI subject (tools/mri_subject.py) with an MHR clip.
Per frame: AP bone radiograph | coronal MRI | sagittal MRI.  Warped MRI intensity, not HU.
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/render_mri.py CLIP.npz OUTDIR [--only i,j]"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
from mhr_clip import load_clip, prepare


def setup():
    import onnxruntime as ort
    from ct_warp import CTWarper
    from benchmark_versions import PerStructure
    bv.REST = np.load(os.environ["REST"])["verts"].astype(float)
    bv.SESS = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    bv.W = CTWarper(os.environ["SUBJECT"], os.environ["CORR"], os.environ["ATLAS"], bv.REST)
    bv.PS = PerStructure(bv.W.atlas, bv.REST)
    W = bv.W
    spine = np.vstack([W.v_ct[n] for n in W.names if n.startswith("vertebrae_")])
    brain = W.v_ct["skull"]; pelvis = np.vstack([W.v_ct[n] for n in W.names if n.startswith(("hip_", "sacrum"))])
    bv.CFG = {"c": spine.mean(0), "y": (brain[:, 1].min() - 60, pelvis[:, 1].max() + 60)}


def win(a, lo, hi):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def frame(p, ref="vertebrae_T8"):
    import render_radiology as rr
    W, CFG, PW = bv.W, bv.CFG, bv.PW
    skin = bv.SESS.run(None, {"mhr_params": p[None].astype(np.float32), "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = bv.posed_for("V4", W.atlas, skin, bv.PS)
    s_, R_, t_ = W.T[ref]
    cr = (((posed[ref] - t_) @ R_) / s_).mean(0)
    c, (y0, y1) = CFG["c"], CFG["y"]
    centre = np.array([c[0], (y0 + y1) / 2, c[2]])
    fld = W.field(posed, skin, structures=True)
    region = (np.array([c[0] - 260, y0, c[2] - 560.0]), np.array([c[0] + 260, y1, c[2] + 220.0]))
    vol, bone, axes, st = W.warped_volume(fld, step=float(os.environ.get("CTW_STEP", 2.0)), region=region, soft_bone=True)
    L = rr.radiograph(vol, bone, 2, st)
    ap = Image.fromarray(rr.unsharp(rr.window(L.T, 0.0, 20.0, 0.75), 2.2, 1.0)).convert("RGB")
    ap = ap.resize((PW, int(ap.height * PW / ap.width)), Image.BICUBIC)
    ims = []
    for ax, pos, a1, a2, cc in ((2, cr[2] - 55.0, 0, 1, centre), (0, cr[0], 2, 1, centre - np.array([0, 0, 100.0]))):
        P, shape = W.plane(ax, pos, a1, a2, cc, 260.0, (y1 - y0) / 2, 2.0)
        mri = W.sample(W.hu_c, W.source_positions(fld, P)).reshape(shape).astype(np.float32)
        g = rr.unsharp(rr.window(mri, 0, 650), 1.2, 0.5)
        im = Image.fromarray(g).convert("RGB")
        ims.append(im.resize((PW, int(im.height * PW / im.width)), Image.LANCZOS))
    return [ap] + ims


if __name__ == "__main__":
    a = sys.argv[1:]
    arr, fps, source = load_clip(a[0]); P = prepare(arr)
    out = a[1]; os.makedirs(out, exist_ok=True)
    bv.PW = int(os.environ.get("PW", 300))
    idx = [int(x) for x in a[a.index("--only") + 1].split(",")] if "--only" in a else range(len(P))
    setup()
    for i in idx:
        cols = frame(P[i])
        H = max(c.height for c in cols); gap = 6
        cv = Image.new("RGB", (len(cols) * (bv.PW + gap), H + 50), (0, 0, 0)); d = ImageDraw.Draw(cv)
        d.text((8, 6), f"MRI (TotalSegmentator-MRI s0175)  {source[:60]}  frame {i + 1}/{len(P)}", fill=(255, 220, 120))
        for k, (im, n) in enumerate(zip(cols, ["AP bone radiograph", "coronal MRI", "sagittal MRI"])):
            cv.paste(im, (k * (bv.PW + gap), 44)); d.text((k * (bv.PW + gap) + 4, 28), n, fill=(235, 235, 235))
        cv.save(f"{out}/{i:03d}.png"); print("frame", i, flush=True)
