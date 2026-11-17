"""Same patient, CT + PET (autoPET, same-session scanner registration), driven by an MHR clip.
Per frame: AP bone radiograph | coronal CT (soft tissue) | coronal PET/CT fusion (SUV 0-8) | sagittal PET/CT fusion.
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/render_petct.py CLIP.npz OUTDIR [--only i,j]   (SUBJECT holds suv.nii.gz)"""
import os, sys
import numpy as np, nibabel as nib
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
from mhr_clip import load_clip, prepare

SUV = None


def setup():
    global SUV
    import onnxruntime as ort
    from ct_warp import CTWarper
    from benchmark_versions import PerStructure
    bv.REST = np.load(os.environ["REST"])["verts"].astype(float)
    bv.SESS = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
    W = bv.W = CTWarper(os.environ["SUBJECT"], os.environ["CORR"], os.environ["ATLAS"], bv.REST)
    bv.PS = PerStructure(W.atlas, bv.REST)
    spine = np.vstack([W.v_ct[n] for n in W.names if n.startswith("vertebrae_")])
    pelvis = np.vstack([W.v_ct[n] for n in W.names if n.startswith(("hip_", "sacrum"))])
    bv.CFG = {"c": spine.mean(0), "y": (spine[:, 1].min() - 90, pelvis[:, 1].max() + 60)}
    SUV = (np.asarray(nib.load(os.environ["SUBJECT"] + "/suv.nii.gz").dataobj).astype(np.float32) * W.keep)


def hot(s, vmax=8.0):
    t = np.clip(s / vmax, 0, 1)
    return np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1)


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
    vol, bone, axes, st = W.warped_volume(fld, step=float(os.environ.get("CTW_STEP", 3.0)), region=region, soft_bone=True)
    L = rr.radiograph(vol, bone, 2, st)
    ap = Image.fromarray(rr.unsharp(rr.window(L.T, 0.0, 20.0, 0.75), 2.2, 1.0)).convert("RGB")
    ap = ap.resize((PW, int(ap.height * PW / ap.width)), Image.BICUBIC)
    out = [ap]
    views = ((2, cr[2] - 55.0, 0, 1, centre), (0, cr[0], 2, 1, centre - np.array([0, 0, 100.0])))
    for k, (ax, pos, a1, a2, cc) in enumerate(views):
        P, shape = W.plane(ax, pos, a1, a2, cc, 260.0, (y1 - y0) / 2, 2.0)
        src = W.source_positions(fld, P)
        ct = W.sample(W.hu_c, src).reshape(shape).astype(np.float32)
        pet = W.sample(SUV, src, cval=0).reshape(shape).astype(np.float32)
        g = rr.unsharp(rr.window(ct, 40 - 175, 40 + 175), 1.4, 0.6).astype(np.float32) / 255.0
        h = hot(pet); a = np.clip(pet / 3.0, 0, 1)[..., None] * 0.85            # fusion: PET opaque above SUV ~3
        fused = (g[..., None].repeat(3, -1) * (1 - a) + h * a)
        ims = ([Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")] if k == 0 else []) + [Image.fromarray((np.clip(fused, 0, 1) * 255).astype(np.uint8))]
        out += [im.resize((PW, int(im.height * PW / im.width)), Image.LANCZOS) for im in ims]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    arr, fps, source = load_clip(a[0]); P = prepare(arr)
    out = a[1]; os.makedirs(out, exist_ok=True)
    bv.PW = int(os.environ.get("PW", 300))
    idx = [int(x) for x in a[a.index("--only") + 1].split(",")] if "--only" in a else range(len(P))
    setup()
    names = ["AP bone radiograph", "coronal CT", "coronal PET/CT (SUV)", "sagittal PET/CT (SUV)"]
    for i in idx:
        cols = frame(P[i]); H = max(c.height for c in cols); gap = 6
        cv = Image.new("RGB", (len(cols) * (bv.PW + gap), H + 50), (0, 0, 0)); d = ImageDraw.Draw(cv)
        d.text((8, 6), f"PET/CT (autoPET, same patient)  {source[:60]}  frame {i + 1}/{len(P)}", fill=(255, 220, 120))
        for k, (im, n) in enumerate(zip(cols, names)):
            cv.paste(im, (k * (bv.PW + gap), 44)); d.text((k * (bv.PW + gap) + 4, 28), n, fill=(235, 235, 235))
        cv.save(f"{out}/{i:03d}.png"); print("frame", i, flush=True)
