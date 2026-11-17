"""Put the cinematic stills into the site: usage: python experiments/add_cinematic.py WORKDIR SITEDIR
Reads WORKDIR/cine/<set>/<clip>/<frame>.png (set: trio | petct | mri | ct) and adds `cine` to the matching docs/data/motions.json entry."""
import os, sys, json, glob
from PIL import Image
work, site = sys.argv[1:3]
mj = f"{site}/data/motions.json"; meta = json.load(open(mj))
for m in meta:
    cid = m["id"]; mod = m.get("modality", "ct")
    clip = cid.split("-", 1)[1] if cid.split("-", 1)[0] in ("mri", "petct", "trio") else cid
    fs = [f for f in glob.glob(f"{work}/cine/{mod}/{clip}/*.png") if not os.path.basename(f).startswith("raw_")]
    if not fs: continue
    dst = f"media/cine-{cid}.jpg"
    Image.open(fs[0]).convert("RGB").save(f"{site}/{dst}", quality=90, optimize=True); m["cine"] = dst
json.dump(meta, open(mj, "w"), indent=1); print(sum("cine" in m for m in meta), "entries with a cinematic still")
