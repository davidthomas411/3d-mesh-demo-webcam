"""PET DICOM series -> SUV(bw) volume resampled onto a CT NIfTI's grid (same patient, same study: the scanner's own registration).
usage: python tools/pet_to_ct_grid.py PET_DICOM_DIR CT.nii.gz OUT.nii.gz"""
import sys, glob, datetime
import numpy as np, pydicom, nibabel as nib
from scipy.ndimage import map_coordinates

def t(s):
    s = str(s).split(".")[0].ljust(6, "0"); return datetime.timedelta(hours=int(s[:2]), minutes=int(s[2:4]), seconds=int(s[4:6]))

def main(pdir, ctf, out):
    ds = [pydicom.dcmread(f) for f in glob.glob(pdir + "/**/*.dcm", recursive=True)]
    ds = [d for d in ds if d.Modality == "PT"]
    d0 = ds[0]; iop = np.array(d0.ImageOrientationPatient, float); rowd, cold = iop[:3], iop[3:]; nrm = np.cross(rowd, cold)
    ds.sort(key=lambda d: float(np.dot(nrm, d.ImagePositionPatient)))
    r = d0.RadiopharmaceuticalInformationSequence[0]
    dose = float(r.RadionuclideTotalDose); hl = float(r.RadionuclideHalfLife)
    dt = (t(d0.SeriesTime if "SeriesTime" in d0 else d0.AcquisitionTime) - t(r.RadiopharmaceuticalStartTime)).total_seconds()
    if dt < 0: dt += 86400
    dose_now = dose * 2 ** (-dt / hl); w = float(d0.PatientWeight) * 1000.0
    print("weight kg", d0.PatientWeight, "dose MBq", dose / 1e6, "decay-corrected dose MBq", round(dose_now / 1e6, 1), "units", d0.Units)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ds], -1)   # (rows, cols, slices)
    if str(d0.Units).upper() == "BQML":
        vol = vol * w / dose_now
    ps = [float(x) for x in d0.PixelSpacing]
    pos = np.array([d.ImagePositionPatient for d in ds], float)
    dz = np.mean(np.diff(pos @ nrm))
    # voxel (i=row index along cols direction? DICOM: row index moves along colDir) -> LPS mm
    A = np.stack([cold * ps[0], rowd * ps[1], nrm * dz], 1); b = pos[0]
    lps2ras = np.diag([-1, -1, 1, 1.0]); aff_pet = lps2ras @ np.vstack([np.hstack([A, b[:, None]]), [0, 0, 0, 1]])
    ct = nib.load(ctf); shape = ct.shape
    inv = np.linalg.inv(aff_pet); o = np.zeros(shape, np.float32)
    ii, jj = np.meshgrid(np.arange(shape[0], dtype=np.float32), np.arange(shape[1], dtype=np.float32), indexing="ij")
    for k in range(shape[2]):
        p = ct.affine @ np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k, np.float32), np.ones(ii.size, np.float32)])
        q = (inv @ p)[:3]
        o[:, :, k] = map_coordinates(vol, q, order=1, mode="constant", cval=0).reshape(shape[:2]).astype(np.float16)
    nib.save(nib.Nifti1Image(o, ct.affine), out)
    print("SUV p50/p99.9/max in body:", np.percentile(o[o > 0.2].astype(float), [50, 99.9]).round(2), float(o.max()))

main(*sys.argv[1:4])
