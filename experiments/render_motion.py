"""Render the motion-between-scans result: [real scan A | A warped through the interpolated motion | real scan B], three planes,
with scan B's bone outline (green) overlaid so convergence is visible.   usage: python experiments/render_motion.py OUTDIR LABEL_A LABEL_B"""
import sys, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi


def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def win(a, lo, hi):
    return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def edge(m):
    return m & ~ndi.binary_erosion(m)


def plane_images(vol, bones, ax_z, cor_y, sag_x, lo, hi, zscale):
    ax = win(vol[ax_z], lo, hi); axb = edge(bones[ax_z])
    cor = win(vol[:, cor_y, :], lo, hi); corb = edge(bones[:, cor_y, :])
    sag = win(vol[:, :, sag_x], lo, hi); sagb = edge(bones[:, :, sag_x])
    out = []
    for img, b, is_ax in ((ax, axb, True), (cor, corb, False), (sag, sagb, False)):
        rgb = np.stack([img] * 3, -1)
        rgb[b] = (60, 255, 90)
        im = Image.fromarray(rgb)
        w = 440
        if is_ax:
            im = im.resize((w, w), Image.LANCZOS)
        else:
            im = im.resize((w, int(img.shape[0] * zscale * w / img.shape[1])), Image.LANCZOS)
        out.append(im)
    return out


if __name__ == "__main__":
    out, la, lb = sys.argv[1:4]
    A = np.load(f"{out}/A.npy").astype(np.float32); B = np.load(f"{out}/B.npy").astype(np.float32)
    bones = np.load(f"{out}/Bbones.npy")
    frames = [np.load(f).astype(np.float32) for f in sorted(glob.glob(f"{out}/frame_*.npy"))]
    lo, hi = np.percentile(A, [3, 99.5])
    zs, ys, xs = np.nonzero(bones)
    ax_z, cor_y, sag_x = A.shape[0] // 2, int(ys.mean()), int(xs.mean())
    zscale = 3.0 / 0.88
    os.makedirs(f"{out}/png", exist_ok=True)
    seq = list(range(len(frames))) + list(range(len(frames) - 2, 0, -1))        # ping-pong A -> B -> A
    cache = {}
    for n, i in enumerate(seq):
        if i not in cache:
            cache[i] = plane_images(frames[i], bones, ax_z, cor_y, sag_x, lo, hi, zscale)
        left = plane_images(A, bones, ax_z, cor_y, sag_x, lo, hi, zscale)
        right = plane_images(B, bones, ax_z, cor_y, sag_x, lo, hi, zscale)
        cols = [left, cache[i], right]
        heights = [sum(im.height for im in c) for c in cols]
        canvas = Image.new("RGB", (440 * 3 + 12, max(heights) + 110), (0, 0, 0))
        d = ImageDraw.Draw(canvas)
        t = i / (len(frames) - 1)
        d.text((10, 8), "Motion between two real scans (same patient, 4D-Lung CBCT)", fill=(240, 240, 240), font=font(20))
        d.text((10, 36), f"interpolated t = {t:0.2f}   |   green = scan B's bone outline (the target)   |   rig model with oracle per-bone poses", fill=(170, 200, 230), font=font(14))
        for c, (cx, title) in enumerate(((0, f"REAL scan A ({la})"), (446, f"A warped through the motion (t = {t:0.2f})"), (892, f"REAL scan B ({lb})"))):
            d.text((cx + 6, 62), title, fill=(255, 220, 120), font=font(15))
            y = 84
            for im in cols[c]:
                canvas.paste(im, (cx, y)); y += im.height
        d.text((10, canvas.height - 22), "Synthetic intermediate frames from a model fitted to the two real scans' bone poses. Not patient care data.", fill=(225, 190, 90), font=font(12))
        canvas.save(f"{out}/png/{n:03d}.png")
    print("frames:", len(seq))
