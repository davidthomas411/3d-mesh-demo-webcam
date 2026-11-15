"""Radiology-style animations from the rig-deformed real CT.

Three products per frame:  skeleton (bone-only AP+lateral radiograph), soft tissue (coronal/sagittal/axial
reformats, soft-tissue window, bone masked, organ contours), and a conventional radiograph (AP+lateral, all tissue).

usage: PYTHONPATH=. python tools/render_radiology.py SUBJECT CORR_NPZ ATLAS ONNX REST_NPZ OUTDIR [--preview a,b,c] [--workers N]
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(__file__))
from ct_warp import CTWarper

# ------------------------------------------------------------------------------------------------ motion
PHASES = [
    ("Spine twist",        lambda u: {6: .10 * np.sin(2 * np.pi * u), 12: .10 * np.sin(2 * np.pi * u), 18: .10 * np.sin(2 * np.pi * u), 21: .10 * np.sin(2 * np.pi * u)}),
    ("Lateral bend",       lambda u: {i: .09 * np.sin(2 * np.pi * u) for i in (8, 14, 19, 22)}),
    ("Shoulder shrug",     lambda u: {31: .32 * np.sin(np.pi * u) ** 2, 41: .32 * np.sin(np.pi * u) ** 2}),
    ("Shoulder protraction", lambda u: {32: .30 * np.sin(np.pi * u) ** 2, 42: .30 * np.sin(np.pi * u) ** 2}),
    ("Forward bend",       lambda u: {i: .12 * (1 - np.cos(2 * np.pi * u)) / 2 for i in (10, 16, 20, 23)}),
]
PER = 28


def params_for(f):
    ph, k = divmod(f, PER)
    name, fn = PHASES[ph]
    p = np.zeros((1, 204), np.float32)
    for i, v in fn(k / PER).items():
        p[0, i] = v
    return name, p


# ------------------------------------------------------------------------------------------------ image helpers
def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def window(img, lo, hi, gamma=1.0):
    return (np.clip((img - lo) / (hi - lo), 0, 1) ** gamma * 255).astype(np.uint8)


def unsharp(u8, sigma=2.0, amount=0.9):
    f = u8.astype(np.float32)
    return np.clip(f + amount * (f - ndi.gaussian_filter(f, sigma)), 0, 255).astype(np.uint8)


def radiograph(vol, mask, axis, step, ref_hi=None, blur=0.6, kappa=1.0):
    """Line integral of attenuation (water = 1) along `axis` -> display image, white = high attenuation."""
    mu = np.clip(vol + 1000.0, 0, None) / 1000.0
    if mask is not None:
        mu = mu * mask
    L = mu.sum(axis) * step / 10.0                            # cm of water-equivalent
    return L


def annotate(im, lines, corners=None, size=15):
    d = ImageDraw.Draw(im)
    f = font(size)
    y = 6
    for t in lines:
        d.text((8, y), t, fill=(215, 215, 215), font=f); y += size + 3
    for (t, xy) in (corners or []):
        d.text(xy, t, fill=(235, 235, 120), font=f)
    return im


def to_rgb(u8):
    return np.stack([u8] * 3, -1)


ORGAN_COL = {"liver": (230, 140, 40), "spleen": (210, 90, 190), "kidney_left": (240, 220, 60), "kidney_right": (240, 220, 60),
             "stomach": (70, 220, 220), "heart": (255, 70, 70), "aorta": (255, 90, 90), "urinary_bladder": (120, 160, 255),
             "pancreas": (150, 255, 120), "colon": (200, 160, 100), "small_bowel": (180, 140, 90), "duodenum": (160, 200, 80)}
for side in ("left", "right"):
    for lobe in ("upper", "lower", "middle"):
        ORGAN_COL[f"lung_{lobe}_lobe_{side}"] = (120, 190, 255)


# ------------------------------------------------------------------------------------------------ rendering
W = None
CFG = {}


def setup(subject, corr, atlas, onnx, rest):
    global W, SESS, REST
    import onnxruntime as ort
    REST = np.load(rest)["verts"].astype(float)
    SESS = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    W = CTWarper(subject, corr, atlas, REST)
    spine = np.vstack([W.v_ct[n] for n in W.names if n.startswith("vertebrae_")])
    c = spine.mean(0)
    CFG["center"] = c
    ymin, ymax = np.vstack([W.v_ct[n] for n in W.names if W.names and n in W.by and n == "skull"])[:, 1].min() - 60, \
        np.vstack([W.v_ct[n] for n in W.names if n.startswith(("hip_", "sacrum"))])[:, 1].max() + 60
    CFG["y"] = (ymin, ymax)
    CFG["label_ids"] = {n: k for k, n in W.label_names.items()}
    t9 = W.v_ct["vertebrae_T9"].mean(0)
    CFG["axial_y"] = t9[1]
    CFG["cor_z"] = c[2] - 55.0                                  # anterior to the vertebral bodies: lungs, heart, liver
    CFG["sag_x"] = c[0]
    CFG["half_x"] = 215.0


def render_frame(f):
    name, p = params_for(f)
    skin = SESS.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
    posed = W.atlas.pose(skin)
    fld = W.field(posed, skin)
    c, (y0, y1), hx = CFG["center"], CFG["y"], CFG["half_x"]
    # ---- projections (2 mm), cropped to the trunk
    region = (np.array([c[0] - hx, y0, c[2] - 190.0]), np.array([c[0] + hx, y1, c[2] + 150.0]))
    vol, bone, axes, st = W.warped_volume(fld, step=2.0, region=region)
    out = {}
    # skeleton: bone-only AP + lateral
    for tag, mask in (("sk", bone), ("rx", None)):
        panels = []
        for ax, label in ((2, "AP"), (0, "LAT")):
            L = radiograph(vol, mask, ax, st)
            img = L.T if ax == 2 else L                         # AP: (x, y) -> (y, x); lateral already (y, z): rows = y, cols = z
            if tag == "sk":
                lo, hi = 0.0, 20.0 if ax == 2 else 16.0
                u8 = window(img, lo, hi, 0.75)
            else:
                lo, hi = 3.0, 38.0 if ax == 2 else 42.0
                u8 = window(img, lo, hi, 1.15)
            u8 = unsharp(ndi.gaussian_filter(u8.astype(np.float32), 0.6).astype(np.uint8), 2.5, 1.1)
            im = Image.fromarray(u8).resize((u8.shape[1] * 2, u8.shape[0] * 2), Image.BICUBIC).convert("RGB")
            ann = [("R", (8, im.height // 2)), ("L", (im.width - 22, im.height // 2))] if ax == 2 else [("A", (8, im.height // 2)), ("P", (im.width - 22, im.height // 2))]
            annotate(im, [label], ann)
            panels.append(im)
        out[tag] = panels
    # ---- reformats at native 1.5 mm (soft tissue window, bone masked, organ contours)
    centre = np.array([c[0], (y0 + y1) / 2, c[2]])
    specs = [("COR", 2, CFG["cor_z"], 0, 1, hx, (y1 - y0) / 2),
             ("SAG", 0, CFG["sag_x"], 2, 1, 130.0, (y1 - y0) / 2),
             ("AX", 1, CFG["axial_y"], 0, 2, hx, 130.0)]
    soft = []
    for label, fixed, pos, hor, ver, hh, hv in specs:
        P, shape = W.plane(fixed, pos, hor, ver, centre, hh, hv, 1.5)
        src = W.source_positions(fld, P)
        hu = W.sample(W.hu_c, src).reshape(shape).astype(np.float32)
        bn = W.sample(W.bone.astype(np.float32), src, cval=0).reshape(shape) > 0.4
        lab = W.sample(W.label.astype(np.float32), src, order=0, cval=0).reshape(shape).astype(np.int32)
        u8 = window(hu, 40 - 175, 40 + 175)
        u8 = unsharp(ndi.gaussian_filter(u8.astype(np.float32), 0.5).astype(np.uint8), 1.5, 0.6)
        rgb = to_rgb(u8)
        rgb[bn] = (38, 38, 44)                                # bone masked (not removed from the physics, hidden for this domain)
        for nm, col in ORGAN_COL.items():
            k = CFG["label_ids"].get(nm)
            if k is None:
                continue
            m = lab == k
            if m.any():
                edge = m & ~ndi.binary_erosion(m)
                rgb[edge] = col
        im = Image.fromarray(rgb).resize((int(rgb.shape[1] * 1.45), int(rgb.shape[0] * 1.45)), Image.LANCZOS)
        ann = ([("R", (8, im.height // 2)), ("L", (im.width - 22, im.height // 2))] if label != "SAG"
               else [("A", (8, im.height // 2)), ("P", (im.width - 22, im.height // 2))])
        if label == "AX":
            ann += [("A", (im.width // 2, 6)), ("P", (im.width // 2, im.height - 22))]
        annotate(im, [{"COR": "Coronal", "SAG": "Sagittal", "AX": "Axial (T9)"}[label]], ann)
        soft.append(im)
    out["st"] = soft
    return f, name, out


def compose(panels, title, subtitle, note, height=None):
    h = max(p.height for p in panels)
    gap = 6
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    head = 56
    foot = 40
    canvas = Image.new("RGB", (w, h + head + foot), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, head + (h - p.height) // 2)); x += p.width + gap
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), title, fill=(240, 240, 240), font=font(22))
    d.text((10, 34), subtitle, fill=(170, 200, 230), font=font(15))
    d.text((10, h + head + 6), note[0], fill=(225, 190, 90), font=font(13))
    d.text((10, h + head + 22), note[1], fill=(140, 140, 140), font=font(12))
    return canvas


NOTE = ("Synthetic: the atlas subject's CT deformed by the anatomy rig. Illustrative; not patient data; not for clinical use.",
        "Displacement = rig motion interpolated by inverse-distance weighting. No sliding, breathing or gas modelled. Arms and couch removed.")


def job(f):
    f, name, out = render_frame(f)
    d = CFG["outdir"]
    compose(out["sk"], "SKELETON domain", f"{name}   |   bone-only radiograph (AP, lateral)   |   frame {f}", NOTE).save(f"{d}/sk_{f:03d}.png")
    compose(out["st"], "SOFT-TISSUE domain", f"{name}   |   soft-tissue window W350/L40, bone masked, organ outlines   |   frame {f}", NOTE).save(f"{d}/st_{f:03d}.png")
    compose(out["rx"], "RADIOGRAPH (all tissue)", f"{name}   |   AP and lateral projection of the deformed CT   |   frame {f}", NOTE).save(f"{d}/rx_{f:03d}.png")
    return f


if __name__ == "__main__":
    a = sys.argv[1:]
    subject, corr, atlas, onnx, rest, outdir = a[:6]
    os.makedirs(outdir, exist_ok=True)
    CFG["outdir"] = outdir
    t0 = time.time()
    setup(subject, corr, atlas, onnx, rest)
    print("setup", round(time.time() - t0), "s", flush=True)
    frames = list(range(len(PHASES) * PER))
    workers = 3
    if "--workers" in a:
        workers = int(a[a.index("--workers") + 1])
    if "--preview" in a:
        frames = [int(x) for x in a[a.index("--preview") + 1].split(",")]
        workers = 1
    t0 = time.time()
    if workers == 1:
        for f in frames:
            job(f); print("frame", f, round(time.time() - t0), "s", flush=True)
    else:
        with Pool(workers) as pool:
            for k, f in enumerate(pool.imap_unordered(job, frames)):
                if k % 10 == 0:
                    print("done", k + 1, "/", len(frames), round(time.time() - t0), "s", flush=True)
    print("finished", round(time.time() - t0), "s")
