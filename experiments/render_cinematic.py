"""Cinematic (path-traced) stills of a posed volume.
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. [MRI=..] python experiments/render_cinematic.py CLIP.npz OUTDIR --frames 0,8 [--mode ct|petct|mri|trio] [--spp 64] [--size 480x720] [--step 3]
The subject folder is the one render_petct.py / render_mri.py use (suv.nii.gz beside ct.nii.gz for PET)."""
import os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import benchmark_versions as bv
import render_petct as rp
import cinematic as cr
from mhr_clip import load_clip, prepare


def get_volume(p, step, mode):
    W, CFG = bv.W, bv.CFG
    skin = bv.SESS.run(None, {"mhr_params": p[None].astype(np.float32), "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = bv.posed_for("V4", W.atlas, skin, bv.PS)
    c, (y0, y1) = CFG["c"], CFG["y"]
    fld = W.field(posed, skin, structures=True)
    region = (np.array([c[0] - 300, y0 - 40, c[2] - 300.0]), np.array([c[0] + 300, y1 + 20, c[2] + 200.0]))
    extras = {}
    if mode in ("petct", "trio"): extras["suv"] = rp.SUV
    if mode == "trio" and rp.MRI is not None: extras["mri"] = rp.MRI[0]
    if mode == "mri":                                          # label masks (0/1 on the MRI grid) warped with the same field
        import anatomy.atlas_build as ab
        ids = W.label_names
        isb = np.isin(W.label, [k for k, n in ids.items() if ab.is_bone(n) and n != "skull" and not n.startswith("humerus")]).astype(np.float32)   # skull is a brain-shell stand-in, arms float free in this MRI
        iso = ((W.label > 0) & (isb == 0) & ~np.isin(W.label, [k for k, n in ids.items() if n == "skull" or n.startswith("humerus")])).astype(np.float32)
        extras["bone"], extras["organ"] = isb, iso
    vol, bone, axes, st = W.warped_volume(fld, step=step, region=region, soft_bone=True, extras=extras)
    return vol, W.extra_out, axes


if __name__ == "__main__":
    a = sys.argv[1:]
    arr, fps, source = load_clip(a[0]); P = prepare(arr)
    out = a[1]; os.makedirs(out, exist_ok=True)
    frames = [int(x) for x in a[a.index("--frames") + 1].split(",")] if "--frames" in a else [len(P) // 2]
    if "--auto" in a:                                       # the most extreme pose of the clip: largest distance from the clip's median pose
        frames = [int(np.argmax(np.linalg.norm(P - np.median(P, 0), axis=1)))]
    mode = a[a.index("--mode") + 1] if "--mode" in a else "ct"
    spp = int(a[a.index("--spp") + 1]) if "--spp" in a else 64
    Wd, Hd = [int(x) for x in (a[a.index("--size") + 1] if "--size" in a else "480x720").split("x")]
    step = float(a[a.index("--step") + 1]) if "--step" in a else 3.0
    az = float(a[a.index("--az") + 1]) if "--az" in a else 0.0
    rp.setup()
    for i in frames:
        t0 = time.time()
        vol, ex, axes = get_volume(P[i], step, mode)
        if mode == "mri":
            sig, col, emis = cr.tf_mri(vol, float(np.percentile(vol[vol > 0], 99)), bone=ex["bone"], organ=ex["organ"])
            nrm, gm = cr.normals_from(ex["bone"] * 1000 + 300 * ex["organ"] + np.clip(vol, 0, 600) * 0.3)
        else:
            sig, col, emis = cr.tf_ct(vol, suv=ex.get("suv"), mri=ex.get("mri"), mri_scale=(rp.MRI[1] if rp.MRI else 1.0))
            nrm, gm = cr.normals_from(np.clip(vol, -300, 1500))
        t1 = time.time()
        # camera frames the body: target = centre of the dense region
        dense = np.argwhere(sig * step > 0.02)
        lo, hi = dense.min(0), dense.max(0)
        tgt = (lo + hi) / 2.0 * step
        hh = 0.5 * (hi[1] - lo[1]) * step * 1.08
        img = cr.render(sig, col, emis, nrm, gm, step, W=Wd, H=Hd, spp=spp, azimuth_deg=az, target=tgt, half_height_mm=hh, seed=17 + i)
        rgb = cr.tonemap(img)
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(f"{out}/raw_{i:03d}.png")
        den = np.clip(cr.denoise(rgb.astype(np.float32)), 0, 1)
        Image.fromarray((den * 255).astype(np.uint8)).save(f"{out}/{i:03d}.png")
        print(f"frame {i}: volume {vol.shape} warp {t1 - t0:.0f}s render {time.time() - t1:.0f}s", flush=True)
