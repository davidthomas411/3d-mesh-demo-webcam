"""Convert one study of the ENHANCE.PET dataset (whole-body FDG PET/CT with CT-derived labels; AWS Open Data bucket
enhance-pet-1-6k; licence per originating site, the autoPET part is CC BY-NC 4.0) into the folder anatomy.atlas_build and
tools/build_ct_volume.py read:

    OUT/ct.nii.gz                    CT, resampled to an isotropic grid
    OUT/suv.nii.gz                   PET on the same grid, in SUV (body weight)
    OUT/segmentations/NAME.nii.gz    one mask per structure, named as TotalSegmentator names them

RAW holds the study's files as downloaded: CT.nii.gz, PT.nii.gz and the label volumes Cardiac, Muscles, Organs, Peripheral-Bones,
Ribs, Vertebrae (.nii.gz). The PET is in Bq/ml, decay-corrected to the start of the scan, so
SUV = activity * weight / (injected dose decayed to the scan start); dose, weight and the two times are in PT-details.xlsx.

Bones below the knee and beyond the elbow are left out of the masks: the rig has no segments for them (the SKEL skeleton does),
and they stay in the CT image itself.

usage: python tools/enhance_subject.py RAW OUT --weight-kg 96 --dose-mbq 320 --uptake-min 60.3 [--voxel-mm 1.5]"""
import argparse, os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

LABELS = {
    "Cardiac": {1: "heart", 2: "heart", 3: "heart", 4: "heart", 5: "heart", 6: "aorta", 7: "iliac_artery_left", 8: "iliac_artery_right",
                9: "iliac_vena_left", 10: "iliac_vena_right", 11: "inferior_vena_cava", 12: "portal_vein_and_splenic_vein", 13: "pulmonary_artery"},
    "Muscles": {1: "autochthon_left", 2: "autochthon_right", 3: "gluteus_maximus_left", 4: "gluteus_maximus_right", 5: "gluteus_medius_left",
                6: "gluteus_medius_right", 7: "gluteus_minimus_left", 8: "gluteus_minimus_right", 9: "iliopsoas_left", 10: "iliopsoas_right"},
    "Organs": {1: "adrenal_gland_left", 2: "adrenal_gland_right", 3: "urinary_bladder", 4: "brain", 5: "gallbladder", 6: "kidney_left",
               7: "kidney_right", 8: "liver", 9: "lung_lower_lobe_left", 10: "lung_lower_lobe_right", 11: "lung_middle_lobe_right",
               12: "lung_upper_lobe_left", 13: "lung_upper_lobe_right", 14: "pancreas", 15: "spleen", 16: "stomach", 17: "thyroid_gland",
               18: "thyroid_gland"},
    "Peripheral-Bones": {3: "clavicula_left", 4: "clavicula_right", 5: "femur_left", 6: "femur_right", 11: "humerus_left", 12: "humerus_right",
                         21: "scapula_left", 22: "scapula_right", 23: "skull"},
    "Ribs": {**{i: f"rib_left_{i}" for i in range(1, 13)}, **{13 + i: f"rib_right_{i}" for i in range(1, 13)}, 27: "sternum"},
    "Vertebrae": {**{i: f"vertebrae_C{i}" for i in range(1, 8)}, **{7 + i: f"vertebrae_T{i}" for i in range(1, 13)},
                  **{19 + i: f"vertebrae_L{i}" for i in range(1, 6)}, 26: "hip_left", 27: "hip_right", 28: "sacrum"},
}
F18_HALF_LIFE_MIN = 109.77


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--weight-kg", type=float, required=True); ap.add_argument("--dose-mbq", type=float, required=True)
    ap.add_argument("--uptake-min", type=float, required=True, help="minutes from injection to the start of the PET series")
    ap.add_argument("--voxel-mm", type=float, default=1.5)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "segmentations"), exist_ok=True)

    ct = nib.load(os.path.join(a.raw, "CT.nii.gz"))
    zoom = np.asarray(ct.header.get_zooms()[:3], float)
    shape = np.ceil(np.asarray(ct.shape) * zoom / a.voxel_mm).astype(int)
    aff = ct.affine.copy(); aff[:3, :3] = ct.affine[:3, :3] @ np.diag(a.voxel_mm / zoom)      # same axes, new spacing
    aff[:3, 3] = ct.affine[:3, :3] @ ((a.voxel_mm / zoom - 1) / 2) + ct.affine[:3, 3]            # voxel centres stay put

    def onto_grid(img, order, cval):
        M = np.linalg.inv(img.affine) @ aff                                                      # new voxel index -> source voxel index
        return ndi.affine_transform(np.asarray(img.dataobj).astype(np.float32), M[:3, :3], M[:3, 3], output_shape=tuple(shape), order=order, cval=cval)

    hu = onto_grid(ct, 1, -1024.0)
    nib.save(nib.Nifti1Image(np.round(hu).astype(np.int16), aff), os.path.join(a.out, "ct.nii.gz"))
    print(f"CT {ct.shape} @ {np.round(zoom, 2)} mm -> {tuple(shape)} @ {a.voxel_mm} mm")

    dose_bq = a.dose_mbq * 1e6 * 0.5 ** (a.uptake_min / F18_HALF_LIFE_MIN)
    k = a.weight_kg * 1000.0 / dose_bq
    suv = onto_grid(nib.load(os.path.join(a.raw, "PT.nii.gz")), 1, 0.0) * k
    nib.save(nib.Nifti1Image(suv.astype(np.float32), aff), os.path.join(a.out, "suv.nii.gz"))
    body = hu > -300
    print(f"PET -> SUV (x {k:.3e}): median in the body {np.median(suv[body]):.2f}, 99.9% {np.percentile(suv[body], 99.9):.1f}, max {suv.max():.1f}")

    n = 0
    for group, names in LABELS.items():
        path = os.path.join(a.raw, group + ".nii.gz")
        if not os.path.exists(path):
            print("missing", path); continue
        lab = np.round(onto_grid(nib.load(path), 0, 0.0)).astype(np.int16)
        for name in sorted(set(names.values())):
            m = np.isin(lab, [i for i, v in names.items() if v == name])
            if m.sum() < 50:
                continue
            nib.save(nib.Nifti1Image(m.astype(np.uint8), aff), os.path.join(a.out, "segmentations", name + ".nii.gz"))
            n += 1
    print(f"{n} structures -> {a.out}/segmentations")


if __name__ == "__main__":
    main()
