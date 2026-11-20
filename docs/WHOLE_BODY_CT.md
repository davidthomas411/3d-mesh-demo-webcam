# Open whole-body CT with arms and legs (for more ambitious motions)

The current CT (TotalSegmentator s1159) is a torso with the arms removed, so arm and leg motion cannot show in the CT panels. Candidates found:

| Dataset | What it has | Licence / access | Status |
|---|---|---|---|
| FDG-PET-CT-Lesions (autoPET), TCIA, DOI 10.7937/gkr0-xv29 | 1,014 CT series, 900 patients (Tubingen). Series named "GK" (Ganzkoerper = whole body): 258 CTs of 700+ slices, 63 of 1000+ (up to 2,651). Typically skull base to mid-thigh, arms usually raised | TCIA API reports CC BY 4.0 for the series (the challenge page says CC BY-NC; check the collection page before publishing) | queried; 1 series downloaded and inspected |
| PSMA-PET-CT-Lesions, TCIA | 378 patients, 597 CT series, up to 642 slices | CC BY 4.0 per the TCIA API | queried only |
| SAROS (Koitka et al., Sci Data 2024), labels on TCIA data | 900 CTs, 299 whole-body, labelled body regions incl. arms and legs, every 5th slice; most whole-body scans truncate at the upper legs | labels CC BY 4.0; some source collections restricted | not downloaded |
| Soft-tissue-Sarcoma, TCIA | 51 patients, CT of the extremities (legs) | CC BY 3.0 | queried only |
| NLM Visible Human (CT) | whole-body axial CT at 1 mm, 512 x 512 (cadavers) | free with NLM acknowledgement; not in the TCIA API | not tried |

Inspected sample (autoPET, `PETCT_0beb67c923`, 978 slices, 0.86 x 0.86 x 1.0 mm, `docs/whole_body_ct_sample_overview.png`): it runs from the upper thorax to the mid-thigh; **no head**, arms raised above the head and cut off at the top of the scan,
both femora to mid-thigh. So it gives legs and (partly) raised arms but not a full avatar. Next steps: inspect the longer "WF" series for ones that include the head, choose an arms-down or arms-by-side case if one exists, run TotalSegmentator, and
rebuild the atlas (the pipeline in `anatomy/atlas_build.py` assumes the MHR rest frame, so arm/leg bones need limb pose normalisation like the current humerus/femur handling).

## HIT (MPI-IS, CVPR 2024): whole-body MRI with tissue segmentations and SMPL fits
https://hit.is.tue.mpg.de/ . Full-body MRI of about 440 people (paper: 260 female, 182 male; the poster lists 241 and 157), segmented into subcutaneous fat, lean tissue (muscle and organs) and long bones
(arms, legs, pelvis), with an SMPL body registered to each surface in two steps that account for lying down. Download needs a registered login and a licence acceptance; I could not read the terms or the file
format (login wall), so check them before relying on it. Useful here for: limb bones and arm/leg articulation, population statistics of internal tissue against body shape, and the learned correction for the
soft-tissue deformation of lying down (our supine CT vs a standing avatar). Not a replacement for CT: it is MRI (no HU for radiographs), tissue classes are coarse (no vertebra-level or organ-level meshes),
and it is SMPL, not MHR (a conversion exists in Fast SAM 3D Body's `mhr2smpl`).
