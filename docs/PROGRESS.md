# Progress log

Newest first. Each entry says what changed, what was measured, and what is still wrong. Commit hashes are in this repo's `claude/femto-bolt-mesh-anatomy-x2x9wl`
branch (the mobile repo carries the shipped atlas and `anatomy.js`). Evidence and numbers live in `docs/BENCHMARK.md`; the plan is `docs/PATIENT_RIG_PLAN.md`.

## Full kinematic tree (in progress)
Goal: joint coordinates (angles/translations inside limits) are the unknowns solved against camera or image data, bones are connected by construction,
plus a penetration-resolving step; `experiments/joint_audit.py` is its regression test. Design and status: `docs/KINEMATIC_TREE.md`.

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
