"""The linac's own gantry angle, read from a video of its console, and matched in time to a gantry recording.

A camera (here a GoPro, upside down) films the console page that shows "Gantry Angle ... Actual". The number is read in
every frame: the field is cropped, its characters are separated, and each is matched against ten digit shapes learnt from
the video itself (the shapes are clustered; which cluster is which digit is given once, by eye, in LABELS).

    python tools/gantry_console.py read VIDEO OUT_DIR            crop the field from every frame (slow: decodes the video)
    python tools/gantry_console.py digits OUT_DIR                cluster the digit shapes -> glyphs.jpg to label
    python tools/gantry_console.py angle OUT_DIR                 console.npz: time (s), angle (degrees, -180..180)
    python tools/gantry_console.py sync OUT_DIR FOLDER GEO_DIR   time offset to the recording, and the comparison
"""
import json
import os
import sys

import numpy as np

# where the number is, for this video: region of the raw frame, then (after turning it the right way up) the number
FIELD = dict(raw=(200, 1100, 1480, 1720), turn=180, number=(150, 520, 35, 150))
LABELS = "9538174260"                        # the ten largest clusters of glyphs.jpg, in order


def glyphs(g):
    """Characters of one field picture, left to right: [(kind 'd' digit / '.' point, 24x32 image, x)]."""
    import cv2
    m = (g < np.median(g) - 40).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if h >= 34 and w >= 8 and a > 150:
            parts = int(round(w / 38.0)) if w > 52 else 1       # digits touching: split evenly
            for q in range(parts):
                xa, xb = x + q * w // parts, x + (q + 1) * w // parts
                out.append(("d", cv2.resize(m[y:y + h, xa:xb].astype(np.float32), (24, 32)), xa))
        elif 4 <= h <= 20 and 4 <= w <= 20 and a >= 12 and y > g.shape[0] * 0.45:
            out.append((".", None, x))
    out.sort(key=lambda t: t[2])
    return out


def unit(im):
    v = im - im.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def cmd_read(video, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    X0, X1, Y0, Y1 = FIELD["raw"]; x0, x1, y0, y1 = FIELD["number"]
    G = np.zeros((n, y1 - y0, x1 - x0), np.uint8); k = 0
    while k < n:
        ok, im = cap.read()
        if not ok:
            break
        c = im[Y0:Y1, X0:X1]
        if FIELD["turn"] == 180:
            c = cv2.rotate(c, cv2.ROTATE_180)
        G[k] = cv2.cvtColor(c[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY); k += 1
    np.savez_compressed(os.path.join(out_dir, "field.npz"), G=G[:k], fps=cap.get(cv2.CAP_PROP_FPS))
    print("frames", k)


def cmd_digits(out_dir):
    import cv2
    G = np.load(os.path.join(out_dir, "field.npz"))["G"]
    cl = []
    for i in range(0, len(G), 3):
        for kind, im, x in glyphs(G[i]):
            if kind != "d":
                continue
            v = unit(im); best, bs = None, 0.80
            for c in cl:
                s = float((unit(c[0] / c[1]) * v).sum())
                if s > bs:
                    best, bs = c, s
            if best is None:
                cl.append([im.copy(), 1])
            else:
                best[0] += im; best[1] += 1
    cl.sort(key=lambda c: -c[1])
    tiles = []
    for k, c in enumerate(cl[:20]):
        t = cv2.resize((255 * (1 - c[0] / c[1])).astype(np.uint8), (72, 96), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 22, 2, 4, 4, cv2.BORDER_CONSTANT, value=200)
        cv2.putText(t, "%d:%d" % (k, c[1]), (2, 16), 0, 0.42, 0, 1); tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.full_like(tiles[0], 200))
    cv2.imwrite(os.path.join(out_dir, "glyphs.jpg"), np.vstack([np.hstack(tiles[r:r + 10]) for r in range(0, len(tiles), 10)]))
    np.savez(os.path.join(out_dir, "templates.npz"), T=np.array([c[0] / c[1] for c in cl[:10]]))
    print("clusters (glyphs.jpg; set LABELS to the digits of the first ten):", [c[1] for c in cl[:20]])


def cmd_angle(out_dir):
    d = np.load(os.path.join(out_dir, "field.npz")); G, fps = d["G"], float(d["fps"])
    T = np.array([unit(t) for t in np.load(os.path.join(out_dir, "templates.npz"))["T"]])
    val = np.full(len(G), np.nan)
    for i in range(len(G)):
        s, ok = "", True
        for kind, im, x in glyphs(G[i]):
            if kind == ".":
                s += "."
                continue
            sc = (T * unit(im)).sum((1, 2)); k = int(np.argmax(sc))
            if sc[k] < 0.75:
                ok = False
                break
            s += LABELS[k]
        if ok and s.count(".") == 1 and 3 <= len(s) <= 5 and len(s.split(".")[1]) == 1 and s[0] != ".":
            v = float(s)
            if 0 <= v < 360:
                val[i] = v
    read = np.isfinite(val)
    a = np.where(val > 180, val - 360, val)
    # a misread shows as a value far from both neighbours: drop it
    for i in np.where(read)[0]:
        nb = a[max(0, i - 4):i + 5]; nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and abs(a[i] - np.median(nb)) > 3:
            a[i] = np.nan
    ok = np.isfinite(a)
    t = np.arange(len(G)) / fps
    ch = np.where(np.abs(np.diff(a[ok])) > 0)[0]
    info = {"frames": int(len(G)), "read": int(ok.sum()), "not_read": int((~ok).sum()), "angle_min": float(np.nanmin(a)), "angle_max": float(np.nanmax(a)),
            "display_updates_per_s_while_moving": round(float(1 / np.median(np.diff(t[ok][ch]))), 1) if len(ch) > 10 else None}
    np.savez(os.path.join(out_dir, "console.npz"), t=t[ok], angle=a[ok])
    print(json.dumps(info, indent=1))
    return info


def cmd_sync(out_dir, folder, geo_dir):
    """Time offset between the console video and the recording, from the angle traces themselves, then how the recording's
    own angles (depth-only, and the cube's) compare with the linac's."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gantry_geometry import GantryGeometry
    c = np.load(os.path.join(out_dir, "console.npz")); tc, ac = c["t"], c["angle"]
    # The console repaints the number about four times a second, so while the gantry moves the picture shows a staircase.
    # A value is fresh at the moment it appears: keep those moments (and both ends of any value held for over 0.6 s) and
    # draw straight lines between them.
    new = np.r_[True, np.diff(ac) != 0]
    start = np.where(new)[0]; end = np.r_[start[1:] - 1, len(ac) - 1]
    keep = set(start.tolist()) | {int(e) for s0, e in zip(start, end) if tc[e] - tc[s0] > 0.6}
    keep = np.array(sorted(keep)); tc, ac = tc[keep], ac[keep]
    rows = np.genfromtxt(os.path.join(folder, "frames.csv"), delimiter=",", names=True, dtype=None, encoding=None)
    tb = (rows["host_us"] - rows["host_us"][0]) / 1e6                   # the recording's frame times, s
    geo = GantryGeometry(os.path.join(geo_dir, "geometry.npz"))
    fr = np.arange(len(tb)); ad = np.array([geo.angle(i) for i in fr])
    moving = np.abs(np.gradient(ad, tb)) > 2.0

    def cost(off, gain=1.0, zero=0.0):
        return np.interp(tb + off, tc, ac, left=np.nan, right=np.nan) - (gain * ad + zero)
    best = None
    for off in np.arange(-60, 60, 0.05):                                # console time = recording time + off
        r = cost(off)[moving]
        if np.isfinite(r).mean() < 0.95:
            continue
        e = np.nanmedian(np.abs(r - np.nanmedian(r)))
        if best is None or e < best[0]:
            best = (e, off)
    off = best[1]
    for _ in range(3):                                                  # refine offset together with gain and zero
        lin = np.interp(tb + off, tc, ac); A = np.c_[ad, np.ones(len(ad))]
        gain, zero = np.linalg.lstsq(A[moving | True], lin, rcond=None)[0]
        offs = np.arange(off - 0.3, off + 0.3, 0.005)
        e = [np.nanmean(np.abs(cost(o, gain, zero)[moving])) for o in offs]
        off = float(offs[int(np.argmin(e))])
    lin = np.interp(tb + off, tc, ac)
    res = lin - (gain * ad + zero)
    out = {"console_time_minus_recording_time_s": round(off, 3),
           "linac_angle_at_recording_frame_0_deg": round(float(np.median(lin[:60])), 2),
           "linac_sweep_deg": [round(float(lin.min()), 1), round(float(lin.max()), 1)],
           "depth_only_angle": {"gain": round(float(gain), 4), "zero_deg": round(float(zero), 2),
                                "residual_rms_deg": round(float(np.sqrt(np.mean(res ** 2))), 2),
                                "residual_rms_moving_deg": round(float(np.sqrt(np.mean(res[moving] ** 2))), 2),
                                "residual_rms_still_deg": round(float(np.sqrt(np.mean(res[~moving] ** 2))), 2),
                                "residual_max_deg": round(float(np.abs(res).max()), 2)}}
    cgp = os.path.join(geo_dir, "cube_geometry.npz")
    if os.path.exists(cgp):
        g = np.load(cgp); f, th, good = g["frames"], g["theta"], g["good"]
        l2 = lin[f]; A = np.c_[th, np.ones(len(th))]
        g2, z2 = np.linalg.lstsq(A[good], l2[good], rcond=None)[0]
        r2 = (l2 - (g2 * th + z2))[good]
        mv = moving[f][good]
        out["cube_angle"] = {"gain": round(float(g2), 4), "zero_deg": round(float(z2), 2), "residual_rms_deg": round(float(np.sqrt(np.mean(r2 ** 2))), 2),
                             "residual_rms_moving_deg": round(float(np.sqrt(np.mean(r2[mv] ** 2))), 2),
                             "residual_rms_still_deg": round(float(np.sqrt(np.mean(r2[~mv] ** 2))), 2), "residual_max_deg": round(float(np.abs(r2).max()), 2)}
    worst = np.argsort(-np.abs(res))[:400]
    out["depth_only_angle"]["frames_more_than_3_deg_off"] = int((np.abs(res) > 3).sum())
    out["depth_only_angle"]["where_worst_frames"] = [int(worst.min()), int(worst.max())]
    for name, m in (("cube_session_0_2500", fr < 2500), ("between_2500_4300", (fr >= 2500) & (fr < 4300)), ("person_session_4300_on", fr >= 4300)):
        out["depth_only_angle"]["rms_deg_" + name] = round(float(np.sqrt(np.mean(res[m] ** 2))), 2)
    extra = {"cube_theta": (lin - z2) / g2} if os.path.exists(cgp) else {}   # the linac's angle in the cube calibration's own scale
    np.savez(os.path.join(geo_dir, "linac_angle.npz"), frames=fr, linac=lin, depth_only=ad, offset_s=off, **extra)
    json.dump(out, open(os.path.join(geo_dir, "linac_angle.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "read":
        cmd_read(sys.argv[2], sys.argv[3])
    elif cmd == "digits":
        cmd_digits(sys.argv[2])
    elif cmd == "angle":
        cmd_angle(sys.argv[2])
    elif cmd == "sync":
        cmd_sync(sys.argv[2], sys.argv[3], sys.argv[4])
