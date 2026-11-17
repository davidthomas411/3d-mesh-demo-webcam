"""DICOM series (CT or MR) -> NIfTI (RAS), optionally resampled onto another NIfTI's grid.
usage: python tools/dicom_to_nii.py DICOM_DIR OUT.nii.gz [--onto CT.nii.gz]"""
import sys, glob
import numpy as np, pydicom, nibabel as nib
from scipy.ndimage import map_coordinates

def load(d):
    ds = [pydicom.dcmread(f) for f in glob.glob(d + "/**/*.dcm", recursive=True)]
    ds = [x for x in ds if "PixelData" in x]
    iop = np.array(ds[0].ImageOrientationPatient, float); rowd, cold = iop[:3], iop[3:]; nrm = np.cross(rowd, cold)
    ds.sort(key=lambda x: float(np.dot(nrm, x.ImagePositionPatient)))
    vol = np.stack([x.pixel_array.astype(np.float32) * float(x.get("RescaleSlope", 1)) + float(x.get("RescaleIntercept", 0)) for x in ds], -1)
    pos = np.array([x.ImagePositionPatient for x in ds], float); ps = [float(v) for v in ds[0].PixelSpacing]
    dz = float(np.mean(np.diff(pos @ nrm))) if len(ds) > 1 else float(ds[0].SliceThickness)
    A = np.stack([cold * ps[0], rowd * ps[1], nrm * dz], 1)
    aff = np.diag([-1, -1, 1, 1.0]) @ np.vstack([np.hstack([A, pos[0][:, None]]), [0, 0, 0, 1]])
    return vol, aff

def resample(vol, aff, onto, cval=0.0):
    ct = nib.load(onto); inv = np.linalg.inv(aff); o = np.full(ct.shape, cval, np.float32)
    ii, jj = np.meshgrid(np.arange(ct.shape[0], dtype=np.float32), np.arange(ct.shape[1], dtype=np.float32), indexing="ij")
    for k in range(ct.shape[2]):
        p = ct.affine @ np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k, np.float32), np.ones(ii.size, np.float32)])
        o[:, :, k] = map_coordinates(vol, (inv @ p)[:3], order=1, mode="constant", cval=cval).reshape(ct.shape[:2])
    return o, ct.affine

if __name__ == "__main__":
    src, out = sys.argv[1:3]; vol, aff = load(src)
    if "--onto" in sys.argv:
        vol, aff = resample(vol, aff, sys.argv[sys.argv.index("--onto") + 1])
    nib.save(nib.Nifti1Image(vol.astype(np.float32), aff), out); print(out, vol.shape)
