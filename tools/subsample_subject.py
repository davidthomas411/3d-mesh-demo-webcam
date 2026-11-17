"""Stride-subsample a CT subject dir (ct.nii.gz, optional suv.nii.gz, segmentations/*.nii.gz) by an integer factor per axis, to fit in RAM.
usage: python tools/subsample_subject.py SRC DST [sx sy sz]   (default 2 2 2)"""
import os, sys, glob
import numpy as np, nibabel as nib

def sub(src, dst, st, mask=False):
    im = nib.load(src); a = np.asanyarray(im.dataobj)[::st[0], ::st[1], ::st[2]]
    aff = im.affine.copy(); aff[:3, :3] = im.affine[:3, :3] @ np.diag(st); aff[:3, 3] = im.affine[:3, 3]
    if mask: a = (a > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(a), aff), dst)
    return int(a.sum()) if mask else 0

src, dst = sys.argv[1:3]; st = [int(x) for x in sys.argv[3:6]] or [2, 2, 2]
os.makedirs(dst + "/segmentations", exist_ok=True)
sub(src + "/ct.nii.gz", dst + "/ct.nii.gz", st)
if os.path.exists(src + "/suv.nii.gz"): sub(src + "/suv.nii.gz", dst + "/suv.nii.gz", st)
for f in sorted(glob.glob(src + "/segmentations/*.nii.gz")):
    n = sub(f, dst + "/segmentations/" + os.path.basename(f), st, True)
    if n == 0: os.remove(dst + "/segmentations/" + os.path.basename(f))
print("done", len(os.listdir(dst + "/segmentations")), "structures")
