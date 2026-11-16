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
