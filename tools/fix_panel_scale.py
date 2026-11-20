"""Rescale the four panels of frames written by experiments/render_clip.py (before its scales were unified) to ONE millimetres-per-pixel scale.
Old scales (px per mm at panel width 300): AP radiograph and coronal 300/520, lateral radiograph 300/780, sagittal 300/560. The lateral view is magnified 1.5x and the sagittal
1.077x, then both are cropped to 300 px around the same body window (z from -361 to +161 mm of the CT-frame centre, as the sagittal plane's own centre). Resampling softens
those two panels slightly; new renders (render_side.py / render_clip.py) are unified at the source.
usage: python tools/fix_panel_scale.py FRAMES_DIR OUT_DIR"""
import os, sys
from PIL import Image

PW, GAP, TOP, FOOT = 300, 6, 64, 22


def fix(im):
    H = im.height - TOP - FOOT
    out = im.copy()
    for k in (1, 3):                                                          # clear the two panels we replace
        out.paste((0, 0, 0), (k * (PW + GAP), TOP, k * (PW + GAP) + PW, im.height - FOOT))
    # lateral radiograph: panel height H*2/3 at 0.385 px/mm -> x1.5 -> 0.577 px/mm; window z in [-361, +161] relative to the CT centre = columns 115..415 of the scaled image
    x = k1 = PW + GAP
    lat = im.crop((x, TOP, x + PW, TOP + round(H * 520 / 780))).resize((round(PW * 1.5), H), Image.LANCZOS).crop((115, 0, 415, H))
    out.paste(lat, (x, TOP))
    # sagittal: panel height H*520/560 at 0.536 px/mm -> x1.077 -> 0.577 px/mm, centred crop to 300 px
    x = 3 * (PW + GAP)
    sag = im.crop((x, TOP, x + PW, TOP + round(H * 520 / 560)))
    sag = sag.resize((round(PW * 560 / 520), H), Image.LANCZOS); c = (sag.width - PW) // 2
    out.paste(sag.crop((c, 0, c + PW, H)), (x, TOP))
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1:3]; os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".png"):
            fix(Image.open(os.path.join(src, f))).save(os.path.join(dst, f))
    print("fixed", len(os.listdir(dst)), "frames")
