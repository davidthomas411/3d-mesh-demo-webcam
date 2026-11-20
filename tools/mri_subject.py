"""Turn a TotalSegmentator-MRI-dataset subject into the CT-shaped subject dir that anatomy.atlas_build reads.

  ct.nii.gz            pseudo-HU: -1000 outside the body, MRI intensity rescaled to 0..1000 inside
                       (only used for the skin surface; render from mri_raw.nii.gz)
  mri_raw.nii.gz       the untouched MRI
  segmentations/*.nii.gz  ground-truth masks, with the single 'vertebrae' class split into levels

The MRI label set has no ribs, sternum, skull, clavicle or scapula, and one 'vertebrae' class. The column is
split between intervertebral discs (largest disc components = real discs, small ones are partial-FOV
fragments) and levels are numbered upward from L5, so level names are an ASSUMPTION, not a measurement.
"""
import argparse, glob, os, shutil
import numpy as np, nibabel as nib, scipy.ndimage as ndi


def body_mask(a):
    s = ndi.gaussian_filter(a, 1.0)
    m = s > 0.08 * np.percentile(a, 99)
    m = ndi.binary_opening(m, iterations=2)
    l, n = ndi.label(m)
    if n > 1:
        m = l == (1 + np.argmax(ndi.sum(m, l, range(1, n + 1))))
    for k in range(m.shape[2]):
        m[:, :, k] = ndi.binary_fill_holes(m[:, :, k])
    return m


def split_vertebrae(v, discs, min_disc_vox=300):
    """Label the vertebra mask with level names, bottom (L5) upward, cut at the midplane between discs."""
    # superior axis = RAS z (checked by caller); one z-index per slice
    l, n = ndi.label(discs)
    zc = sorted(float(np.argwhere(l == i)[:, 2].mean()) for i in range(1, n + 1) if (l == i).sum() >= min_disc_vox)
    zv = np.where(v.any((0, 1)))[0]
    edges = [zv.min() - 1] + [0.5 * (a + b) for a, b in zip(zc[:-1], zc[1:])] + [zv.max() + 1]
    # a vertebra sits between two consecutive discs
    cuts = zc
    names = ["L5", "L4", "L3", "L2", "L1"] + [f"T{k}" for k in range(12, 0, -1)] + [f"C{k}" for k in range(7, 0, -1)]
    z = np.arange(v.shape[2])
    out = {}
    bounds = [zv.min() - 0.5] + cuts + [zv.max() + 0.5]
    for i in range(len(bounds) - 1):
        sl = (z > bounds[i]) & (z <= bounds[i + 1])
        m = v & sl[None, None, :]
        # segment 0 lies below the L5-S1 disc (sacral end): not a mobile level
        if i >= 1 and m.sum() > 200 and i - 1 < len(names):
            out["vertebrae_" + names[i - 1]] = m
    return out


def main(src, dst):
    os.makedirs(dst + "/segmentations", exist_ok=True)
    im = nib.load(src + "/mri.nii.gz")
    assert nib.aff2axcodes(im.affine) == ("R", "A", "S"), nib.aff2axcodes(im.affine)
    a = np.asanyarray(im.dataobj).astype(np.float32)
    bm = body_mask(a)
    hu = np.full(a.shape, -1000, np.int16)
    hu[bm] = np.clip(a[bm] / np.percentile(a[bm], 99) * 1000, 0, 1500).astype(np.int16)
    nib.save(nib.Nifti1Image(hu, im.affine), dst + "/ct.nii.gz")
    shutil.copy(src + "/mri.nii.gz", dst + "/mri_raw.nii.gz")
    for f in sorted(glob.glob(src + "/segmentations/*.nii.gz")):
        n = os.path.basename(f)[:-7]
        m = np.asanyarray(nib.load(f).dataobj) > 0
        if not m.any() or n in ("vertebrae", "intervertebral_discs"):
            continue
        l, k = ndi.label(m)                      # ground truth has stray islands (e.g. femur label in the chest)
        if k > 1:
            m = l == (1 + np.argmax(ndi.sum(m, l, range(1, k + 1))))
        nib.save(nib.Nifti1Image(m.astype(np.uint8), im.affine), f"{dst}/segmentations/{n}.nii.gz")
    # The MRI label set has no skull. The rig needs one rigid head body, so use a 6 mm shell of the brain
    # surface as a stand-in named 'skull'. It is NOT bone and is not a measurement of the cranium.
    bp = f"{dst}/segmentations/brain.nii.gz"
    if os.path.exists(bp):
        b = np.asanyarray(nib.load(bp).dataobj) > 0
        r = max(1, int(round(6.0 / min(np.linalg.norm(im.affine[:3, :3], axis=0)[:2]))))
        shell = b & ~ndi.binary_erosion(b, iterations=r)
        nib.save(nib.Nifti1Image(shell.astype(np.uint8), im.affine), f"{dst}/segmentations/skull.nii.gz")
    v = np.asanyarray(nib.load(src + "/segmentations/vertebrae.nii.gz").dataobj) > 0
    d = np.asanyarray(nib.load(src + "/segmentations/intervertebral_discs.nii.gz").dataobj) > 0
    lv = split_vertebrae(v, d)
    for n, m in lv.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), im.affine), f"{dst}/segmentations/{n}.nii.gz")
    print("levels:", list(lv), "body voxels", int(bm.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); a = ap.parse_args(); main(a.src, a.dst)
