"""Export a rest-pose volume for the browser path tracer (docs/cine/).
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. [MRI=..] python tools/export_cine_volume.py NAME OUTDIR [--step 3]
Writes NAME.json (shape, step_mm, target, half_height) and NAME.bin.gz = float16 HU (x fastest) | uint8 SUV*16 (x fastest) | optional uint8 MRI/p99*255."""
import os, sys, json, gzip
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_versions as bv
import render_petct as rp
import render_cinematic as rc
from mhr_clip import prepare

name, out = sys.argv[1:3]; step = float(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 3.0
os.makedirs(out, exist_ok=True); rp.setup()
p = np.zeros(204, np.float32)
vol, ex, axes = rc.get_volume(p, step, "trio" if rp.MRI is not None else "petct" if rp.SUV is not None else "ct")
dense = np.argwhere(vol > -300); lo, hi = dense.min(0), dense.max(0) + 1
sl = tuple(slice(l, h) for l, h in zip(lo, hi)); vol = vol[sl]
parts = [np.ascontiguousarray(vol.astype(np.float16).transpose(2, 1, 0)).tobytes()]
has = {"suv": "suv" in ex}
if "suv" in ex: parts.append(np.ascontiguousarray(np.clip(ex["suv"][sl] * 16, 0, 255).astype(np.uint8).transpose(2, 1, 0)).tobytes())
if "mri" in ex:
    m = ex["mri"][sl]; has["mri"] = True; parts.append(np.ascontiguousarray(np.clip(m / rp.MRI[1] * 255, 0, 255).astype(np.uint8).transpose(2, 1, 0)).tobytes())
meta = {"name": name, "shape": list(vol.shape), "step_mm": step, "has": has, "origin_mm": [float(axes[a][lo[a]]) for a in range(3)],
        "units": "HU float16, SUV*16 uint8, MRI/p99*255 uint8; arrays x-fastest"}
json.dump(meta, open(f"{out}/{name}.json", "w"))
with gzip.open(f"{out}/{name}.bin.gz", "wb", 6) as f:
    for b in parts: f.write(b)
print(name, vol.shape, f"{os.path.getsize(f'{out}/{name}.bin.gz') / 1e6:.1f} MB", flush=True)
