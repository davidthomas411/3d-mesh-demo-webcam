"""S1 pilot: how well does a rigid SURFACE registration (what SGRT does) locate internal bones, day to day?

Open data: TCIA 4D-Lung (CC BY 3.0). One patient, several treatment days, one breathing phase per day (daily 4D-CBCT, 0% phase).
For each day k versus the first day:
  surface_icp : rigid trimmed-ICP of the anterior skin surface (the part a camera sees)  -> residual of each bone landmark
  bone_rigid  : best single rigid transform of the bone landmarks themselves            -> the floor for ANY rigid method
  The gap between the two, and the residual of bone_rigid, are what a rig that models articulation could still remove.
Bone landmarks = centroids of TotalSegmentator vertebrae/scapulae/sternum that are fully inside the slab on both days
(partial-FOV CBCT: structures cut by the slab edge are skipped because truncation moves their centroid).

usage: python experiments/s1_surface_vs_bone.py WORKDIR [--patient 100_HM10395] [--days 6] [--no-seg]
Requires: SimpleITK, nibabel, scipy, TotalSegmentator (set TOTALSEG_HOME_DIR to a dir containing weights and
config.json with send_usage_stats=false). Nothing here uploads data.
"""
import sys, os, io, json, glob, zipfile, subprocess, urllib.request, urllib.parse, collections
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy.spatial import cKDTree

B = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
BONE_PREFIX = ("vertebrae_T", "scapula_", "sternum")


def get(ep, **kw):
    with urllib.request.urlopen(B + ep + "?" + urllib.parse.urlencode(kw), timeout=300) as r:
        return r.read()


def pick_series(patient, n_days):
    sr = json.loads(get("getSeries", Collection="4D-Lung"))
    cb = [s for s in sr if s["PatientID"] == patient and s["Modality"] == "CT" and s["ImageCount"] <= 60
          and s.get("SeriesDescription", "").strip().endswith("Gated, 0.0%")]
    by_date = {}
    for s in cb:
        by_date.setdefault(s["StudyDate"][:10], s)
    dates = sorted(by_date)
    idx = np.unique(np.linspace(0, len(dates) - 1, n_days).round().astype(int))
    return [(dates[i], by_date[dates[i]]) for i in idx]


def fetch_nifti(series, out):
    if os.path.exists(out):
        return out
    z = zipfile.ZipFile(io.BytesIO(get("getImage", SeriesInstanceUID=series["SeriesInstanceUID"])))
    d = out + ".dicom"
    os.makedirs(d, exist_ok=True)
    z.extractall(d)
    r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(d))
    sitk.WriteImage(r.Execute(), out)
    return out


def segment(nii, outdir):
    if not os.path.exists(outdir) or not os.listdir(outdir):
        subprocess.run(["TotalSegmentator", "-i", nii, "-o", outdir, "--fast", "--device", "cpu", "--quiet"], check=True)
    return outdir


def skin_points(nii, step=3):
    """Anterior skin surface samples (physical LPS mm): first body voxel from the front for each (x, z) column."""
    img = sitk.ReadImage(nii)
    a = sitk.GetArrayFromImage(img).astype(np.float32)               # (z, y, x)
    from skimage.filters import threshold_otsu
    thr = float(threshold_otsu(np.clip(a, -1500, 500)))              # air vs body (CBCT HU are uncalibrated)
    body = a > max(thr, -600)
    pts = []
    for k in range(0, a.shape[0], 1):
        for i in range(0, a.shape[2], step):
            col = np.flatnonzero(body[k, :, i])
            if len(col):
                pts.append((i, col[0], k))                          # LPS: y increases posteriorly, so col[0] is anterior
    idx = np.array(pts, float)
    return np.array([img.TransformContinuousIndexToPhysicalPoint(tuple(p)) for p in idx])


def landmarks(segdir, nii):
    ref = nib.load(nii)
    out = {}
    for f in sorted(glob.glob(f"{segdir}/*.nii.gz")):
        name = os.path.basename(f)[:-7]
        if not name.startswith(BONE_PREFIX):
            continue
        m = np.asanyarray(nib.load(f).dataobj) > 0
        if m.sum() < 800:
            continue
        zs = np.flatnonzero(m.any(axis=(0, 1)))
        if zs.min() == 0 or zs.max() == m.shape[2] - 1:             # cut by the slab: centroid is not reliable
            continue
        ijk = np.argwhere(m).mean(0)
        out[name] = (ref.affine @ np.append(ijk, 1))[:3]            # NIfTI RAS mm (consistent across days)
    return out


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp


def icp(src, dst, iters=40, trim=0.8):
    R, t = np.eye(3), np.zeros(3)
    tree = cKDTree(dst)
    for _ in range(iters):
        p = src @ R.T + t
        d, i = tree.query(p)
        keep = d <= np.quantile(d, trim)
        dR, dt = kabsch(p[keep], dst[i[keep]])
        R, t = dR @ R, dR @ t + dt
    return R, t


def to_ras(pts_lps):
    return pts_lps * np.array([-1.0, -1.0, 1.0])


if __name__ == "__main__":
    work = sys.argv[1]
    patient = sys.argv[sys.argv.index("--patient") + 1] if "--patient" in sys.argv else "100_HM10395"
    n_days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 6
    os.makedirs(work, exist_ok=True)
    days = pick_series(patient, n_days)
    print("days:", [d for d, _ in days], flush=True)
    data = []
    for d, s in days:
        nii = fetch_nifti(s, f"{work}/{patient}_{d}.nii.gz")
        seg = segment(nii, f"{work}/seg_{d}") if "--no-seg" not in sys.argv else f"{work}/seg_{d}"
        data.append((d, to_ras(skin_points(nii)), landmarks(seg, nii)))
        print(d, "skin pts", len(data[-1][1]), "bone landmarks", len(data[-1][2]), flush=True)
    d0, skin0, lm0 = data[0]
    rows = []
    for d, skin, lm in data[1:]:
        shared = sorted(set(lm0) & set(lm))
        if len(shared) < 4:
            print(d, "skipped: only", len(shared), "shared landmarks"); continue
        P = np.array([lm0[n] for n in shared]); Q = np.array([lm[n] for n in shared])
        Rs, ts = icp(skin0, skin)                                    # surface-only rigid registration
        res_surface = np.linalg.norm(P @ Rs.T + ts - Q, axis=1)
        Rb, tb = kabsch(P, Q)                                        # bone-based optimum for any single rigid transform
        res_bone = np.linalg.norm(P @ Rb.T + tb - Q, axis=1)
        raw = np.linalg.norm(P - Q, axis=1)
        shift_s = np.linalg.norm(ts); shift_b = np.linalg.norm(tb)
        rows.append((d, len(shared), raw.mean(), res_surface.mean(), res_surface.max(), res_bone.mean(), res_bone.max()))
        print(f"{d}: {len(shared)} landmarks | no registration {raw.mean():5.1f} | after surface-ICP {res_surface.mean():5.1f} (max {res_surface.max():5.1f})"
              f" | after bone-optimal rigid {res_bone.mean():5.1f} (max {res_bone.max():5.1f})  [mm]", flush=True)
    json.dump(rows, open(f"{work}/s1_results.json", "w"))
