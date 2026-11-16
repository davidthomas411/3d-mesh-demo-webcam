"""MRI gallery: take rendered MRI frames (experiments/render_mri.py), put the original Rocketbox motion beside them, encode video + poster and
add 'mri-<clip>' entries (modality "mri") to docs/data/motions.json.
usage: python experiments/make_gallery_mri.py WORKDIR SITEDIR   (WORKDIR holds clips/, rocketbox/, mri_frames/<clip>/NNN.png)"""
import os, sys, json, subprocess
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gallery import CLIPS, duration
import math

work, site = sys.argv[1:3]
mj = f"{site}/data/motions.json"; meta = {m["id"]: m for m in json.load(open(mj))}
for n, (title, blurb) in CLIPS.items():
    fr = f"{work}/mri_frames/{n}"; z = np.load(f"{work}/clips/{n}.npz"); nf = len(z["mhr_model_params"])
    if not os.path.isdir(fr) or len([f for f in os.listdir(fr) if f.endswith(".png")]) < nf:
        print(n, "frames incomplete, skipped"); continue
    sk, eng = f"{work}/rocketbox/{n}.skeleton.glb", f"{work}/rocketbox/{n}.engine.glb"
    stride = max(1, math.ceil((duration(sk) * 30 + 1) / 16)); orig = f"{work}/mri_with_orig/{n}"
    subprocess.run([sys.executable, f"{ROOT}/tools/rocketbox_original_panel.py", eng, fr, orig, "--stride", str(stride)], check=True)
    mid = f"mri-{n}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", f"{orig}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{site}/media/{mid}.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{orig}/{nf // 2:03d}.png", "-vf", "scale=900:-1", "-q:v", "4", f"{site}/media/{mid}.jpg"], check=True)
    m = dict(meta[n]); m.update({"id": mid, "modality": "mri", "title": title + " (MRI)"})
    meta[mid] = m; print(mid, "done", flush=True)
json.dump(list(meta.values()), open(mj, "w"), indent=1)
