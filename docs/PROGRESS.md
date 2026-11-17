# Progress log

Newest first. Each entry says what changed, what was measured, and what is still wrong. Commit hashes are in this repo's `claude/femto-bolt-mesh-anatomy-x2x9wl`
branch (the mobile repo carries the shipped atlas and `anatomy.js`). Evidence and numbers live in `docs/BENCHMARK.md`; the plan is `docs/PATIENT_RIG_PLAN.md`.

## 2026-11-17 PET/CT: matched CT + FDG PET of one patient driven by the rig
- autoPET patient PETCT_0beb67c923 (TCIA FDG-PET-CT-Lesions): the PET series was downloaded, converted to SUVbw with `tools/pet_to_ct_grid.py` (decay-corrected dose, weight) and resampled onto the CT grid (scanner registration). TotalSegmentator fast mode (CPU, 3 mm, ~3 min) segmented the CT; `tools/subsample_subject.py` strides the subject by 2 to fit 15 GB; `anatomy.atlas_build` built a full rig (shoulders, 24 costovertebral joints).
- Build fixes for a scan with no head and a truncated thorax: skull seating is optional; the scapulothoracic ellipsoid falls back to a bounding-box ellipsoid when the least-squares fit is degenerate.
- `experiments/render_petct.py` warps CT and PET with the same field; all 8 clips are on the Pages site (modality `petct`).
- Matched CT/PET/MRI: TCIA Soft-tissue-Sarcoma has CT, PET and MR for all 51 patients, with MR resampled to the PET frame ("Aligned_T1toPET_BOX"); next step.
- MRI brain contrast: brain median signal is ~470 vs liver ~90 in s0175 (head coil gain and station stitching in a stitched whole-body gradient-echo), so a single fixed window saturates the head.

## 2026-11-16 MRI: the same rig drives a TotalSegmentator-MRI subject
- `tools/mri_subject.py` converts a subject of the TotalSegmentator MRI dataset (Zenodo 11367005, CC BY-NC-SA 2.0) into the CT-shaped folder `anatomy.atlas_build` reads. Subject s0175 (head to mid-thigh, 53 structures) built an atlas (`atlas_build`, `ct_correspondence`) and `experiments/render_mri.py` warps the MRI with the existing CT warper (`docs/mri_rig_demo.png`).
- The MRI label set has no ribs, sternum, clavicle, scapula, skull or per-level vertebrae. Spine levels are cut between the real disc components and numbered up from L5 (an assumption). The 'skull' is a 6 mm shell of the brain surface used as the rigid head body (not bone). Rig joints present: hips only; no shoulder/costovertebral joints.
- Stray label islands (a femur label in the chest) are removed by keeping the largest component per mask. Coronal reformat shows stitching seams between MRI stations (the source is multi-station).
- Gallery: all 8 Rocketbox clips rendered on the MRI (`experiments/make_gallery_mri.py`), shown in a separate MRI section of docs/index.html (`modality: mri` in motions.json).
- `ct_correspondence.py` hardcodes the 140000-triangle budget; build MRI atlases with the default `--tris`.

## Full kinematic tree (in progress)
Goal: joint coordinates (angles/translations inside limits) are the unknowns solved against camera or image data, bones are connected by construction,
plus a penetration-resolving step; `experiments/joint_audit.py` is its regression test. Design and status: `docs/KINEMATIC_TREE.md`.

## Rig-versus-skin "pivot mismatch": it was in the CT mapping, not the rig
- In atlas space the rig agrees with the MHR skin (skin on bone moved with its bone lands a median 0.5-5 mm away for the skull, 4 mm pelvis, 6 mm cervical, 12-15 mm ribs, sternum and spine,
  `experiments/rig_skin_consistency.py`). The 93 mm skull-vs-skin gap at a 100 degree head turn came from how bone motion was transferred to the CT frame: my `bone_ct_positions` moved
  each CT bone about its own centroid (before that, through its own CT-to-atlas similarity), adding (I - Q) times the offset between the CT bone and its atlas counterpart (50-90 mm at 100 degrees).
  Now the CT bone takes the atlas bone's own rigid motion, y = Q W(x) + tau, exactly as skin does. Skull-vs-skin residual in the CT frame: 93 mm to 3.4 mm; all bone groups near the skin median 14.8 mm
  (was 43 mm) at the head-turn frame, 8.2 mm (was 12.6) near rest (`experiments/skin_bone_consistency.py`).
- Also tested and rejected: carrying soft-tissue landmarks by the local rotation (`CTW_RT_SOFT=1`) threw head tissue to the abdomen and doubled held-out error (default off). Skin uses the local rotation
  (`CTW_RT_SKIN`, default on).
- Held-out accuracy at the head-turn frame (build the field from half the landmarks, reconstruct the rest): skin median 3.1 mm / p90 12.0 (was 6.1 / 29.6), soft landmarks median 1.5 / p90 7.4 / p99 16.7 (was 1.7 / 11.2 / 23.2).
- Remaining: ribs, sternum and spine disagree with the skin by 15-25 mm in the thorax (partly real: skin over the shoulder follows the arms, not the ribs); the rig's own pelvis and root motion are not retargeted.

## Motion gallery (docs/)
- `experiments/make_gallery.py` retargets, renders (rig + CT with the original motion beside it), encodes and indexes Rocketbox clips; `docs/index.html` is the player page (data from `docs/data/motions.json`).

## Mushy CT soft tissue: diagnosis and first fix
- Bones render fine; the CT reformats tore (face), smeared (torso) and left black gaps beside bones. Measured instead of guessed:
  1. The soft-tissue warp interpolates landmarks from three sources (skin, bones, organs), and they CONTRADICT each other. Skin within 20 mm of a bone, moved with that bone's rigid motion,
     lands a median 12.6 mm from the posed MHR skin at a near-rest frame and 43 mm at a 100 degree head turn (93 mm on the skull, 40 mm on ribs, 43 mm on the spine). Cause: the rig's bones
     and the MHR skin pivot differently (skull about the atlanto-occipital joint vs skin about MHR's neck; the skin-derived chain vs rigid ribs), so the interpolation must average
     opposite motions. Comparing displacement vectors is the wrong test for rotating bodies; the position test above is the valid one.
  2. A held-out test (build the field from half the skin points, reconstruct the other half) gave median 6.1 mm, p90 29.6 mm, p99 53.5 mm on skin that is only 3.5 mm apart: the field cannot
     even reproduce a smooth surface.
  3. Not the cause: translation-only interpolation (a moving-least-squares rigid field, `method="mls"`, scored the same), organ landmarks (dropping them barely helped), and the frame in which each
     bone's motion is expressed (a global-registration version is in, `bone_ct_positions`, but changed little).
- Fix 1: skin landmarks on bone now follow that bone (`skin_targets`, weight exp(-d^2/2*25 mm^2)), so the soft tissue between bone and skin interpolates between landmark sets that agree where they meet.
  Held-out skin error: median 4.1 mm (was 6.1), p90 18.6 (29.6), p99 36.9 (53.5); mid-thorax 5.9 mm (12.1). Face and jaw no longer shred at a 100 degree head turn; see
  `docs/soft_tissue_warp_before_after.png` (before left, after right; look-around frame 17 above, shrug frame 10 below).
- Still wrong: p90 skin error ~19 mm remains (other contradictions: ribs 40 mm, sternum 31 mm vs skin even near rest); the fix hides the head conflict by following the bone, it does not make
  the rig pivot where the MHR skin does. Fixing the pivot (AO joint and the chain anchoring) is the real repair. The two Rocketbox videos were rendered before this fix.

## Real open mocap through the rig (Rocketbox shrug) and the multi-view capture reuse path
- Microsoft Rocketbox motions (MIT, ungated) retargeted to MHR by direction fitting (0.3 deg mean error) and rendered through skeleton + CT; see `docs/MOCAP_CLIPS.md`.
- Brain now rigid with the skull using the skull's own transform in the CT warp (fixed a lattice artefact in the skull).

## Motion-capture clip pipeline (synthetic clip first)
- Loader, per-frame renderer (AP + lateral radiograph, coronal + sagittal real HU) and a video-to-clip script for SAM 3D Body; see `docs/MOCAP_CLIPS.md`.
- No real SAM 3D Body clip yet: checkpoints are gated (401) and no public MHR sequences were found; the video-to-clip script is untested against the real model.
- Render memory fixed (slabbed `warped_volume`); AP region widened so a head bent forward is not clipped; brain rigid with the skull in the CT warp; side-view renderer
  (`experiments/render_side.py`). Neck bones stay connected through the forward bend (C7/T1 gap 1.1-1.3 mm); face/jaw soft tissue still tears at the peak.

## Scapulothoracic contact (atlas v8)
- Arm-elevation check found the scapula floating off the rib cage when the arm is raised forward (gap 11 mm at rest -> 27 mm at 90 deg flexion).
- Added a contact step: the scapula is swung about its acromioclavicular pin until the mean gap of its rib-facing patch to a thorax ellipsoid is back to
  the rest value, with a term that stops it going deeper than the rest closest point. Numpy (`_glide`) and JS (`glide`) with parity test; atlas stores the patch,
  ellipsoid and rest gaps in the AC joint entries.
- Result: forward flexion gap 12-15 mm (was up to 27 mm). Arm abduction still penetrates the ribs by 2-3 mm at the extremes (the ellipsoid proxy is coarse); not solved.
- Spine range-of-motion audit (`experiments/spine_limits_audit.py`): every per-level rotation in the benchmark motions is far inside assumed limits (max 3.6 deg flexion,
  2.7 deg axial per level), so no level is over limit; the limits are assumed from the classic tables and not verified here.

## 11a8e36 Broken neck: atlas v7 (spine re-joined at group boundaries)
- Cause: each bone group was placed rigidly on its own to fit the skin, leaving the spine disconnected at the boundaries *at rest* (T1/C7 9.2 mm, L1/T12 23.2 mm).
  The change-from-rest audit could not see this.
- Fix: one 18-parameter solve (`tools/close_junctions.py`); junctions now 0.7-1.4 mm against the CT subject's 0.6-1.1 mm, core bones still inside the skin.
- Cost: large group moves (thorax 60 mm) because avatar and CT proportions differ; ball-joint pin test tolerance relaxed 1.5 -> 2.5 mm.
- Picture: `docs/compare_neck_fix.png`.

## ca4b9af Flexible costal cartilage, exact joint audit
- Cartilage was one rigid body, so rib/cartilage junctions opened up to 9 mm. Now a flexible link (ends follow nearest rib and sternum); ported to `anatomy.js`, parity tested.
- The first audit's vertebral and clavicle overlaps (16-18 mm) were artefacts of a rough sign test; the exact-mesh audit replaced it.

## 334108c Ghost skulls in forward bend
- The smooth warp field mapped voxels near the posed head back onto the source skull. Bones now come only from the exact rigid override found from a dense posed-bone tree.

## 983db53 / a4beba2 / b31d24d Rendering fixes
- Soft bone fraction at 2 mm step removes moire ghosting; coronal reformats show real HU (no bone mask); real-scan video had head down (DICOM z index 0 = feet), flipped.

## 31c53f2 Population-consensus respiratory pilot
- 8 patients of 4D-Lung, leave-one-patient-out: population prior 2.86 mm vs 3.76 mm for no motion; fitting a patient scale from <=4 structures was worse than the prior (pilot only).

## a3e01f3 Benchmark and physiologic spine split
- Five stages V0-V4 on one real CT with one fixed motion; V4 shares trunk rotation along the vertebrae by in-vivo per-level data (Fujimori 2012; lumbar values assumed).

## Earlier
Femto Bolt depth fusion, anatomy atlas from TotalSegmentator ground truth, biomechanical rig with joint rules, rig-deformed CT radiology, real-scan motion
experiment (4D-Lung CBCT), patient-rig plan with literature and open-data survey (`3759ffc`..`f143ba0`, `5bd51d0`).

## Known issues and not done
Push to GitHub blocked (403: Claude GitHub App not installed on the webcam repo); work is committed locally and mirrored as a git bundle. Sternoclavicular joint
opens up to ~2.4 mm; scapulothoracic contact, collision resolution and soft-tissue sliding not modelled; breathing soft-tissue model and calibration, JS port of
physio/no-joints modes, head-and-neck open data, GPU port for real-time all outstanding.
