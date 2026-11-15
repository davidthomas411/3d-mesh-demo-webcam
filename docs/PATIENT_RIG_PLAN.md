# Patient-specific, drivable biomechanical rig learned from serial imaging

Status: design + data survey (no model trained yet). Everything marked **[verified]** was retrieved in the session that wrote
this (TCIA catalogue API, PubMed, Consensus). Everything marked **[proposal]** is a design decision, not a result.

## 1. The problem

Images are snapshots and registration is blind to anatomy. We want a model that gives **voxel-to-voxel correspondence across
time points** (planning CT, daily CBCT, mid-treatment, follow-up) by representing each patient as a *drivable rig*: rigid bones,
joints with patient-specific centres/axes/limits, and soft tissue that moves with them. The rig is learned **per patient** from
that patient's own prior scans, on top of a population prior.

## 2. What exists  [verified]

| Piece | Work | What it shows | Gap for us |
|---|---|---|---|
| Articulated skeleton for registration | Walter et al. 2025, *Comput Methods Biomech Biomed Eng Imaging Vis*: "articulated head-and-neck motion model using deep learning-based instance segmentation of skeletal bones" | Auto bone segmentation can build the articulated registration model; CT-CT TRE 1.51 +/- 0.26 mm; TotalSegmentator alone lacked structures the full model needs | Head and neck only; joint model hand-designed; no learning of patient-specific joints across scans |
| Articulated digital twin from CT | Zhang et al. 2026 (arXiv preprint): "Patient-Specific Articulated Digital Twins from a Single Full-Body CT Scan" | SMPL scaffold fitted to CT, bones/organs bound to a rig, retargeted poses; 15.8 +/- 4.0 mm chamfer, 95.9% skeletal enclosure, DRR SSIM 0.872; 3 subjects | Single scan, generic joints, no longitudinal learning, no validation against real second scans |
| Vertebral joint pose statistics | Gastreich de Llanes et al. 2026, *Curr Dir Biomed Eng* | SSM/ASM + kPCA give priors on vertebral joint poses to constrain registration; mean shape error 7.55 mm | Spine only; a prior, not patient-specific |
| Population + patient-specific deformation | Rortveit et al. 2023, *Phys Med Biol* (Bayesian) | Combining population and patient data predicts individual motion with fewer scans (1-3 scans beat either alone) | Organs (rectal wall), no skeleton/joint structure |
| Inter-fraction variation | Pastor-Serrano et al. 2023 (DAM); Smolders et al. 2024 (DiffuseRT) | Generative models of likely anatomical change from the planning image (312 CT pairs / 38 pts; 224+152 CBCTs) | Image-space statistics, no kinematic structure, not per-patient learned |
| Statistical respiratory motion | Ehrhardt et al. 2010 (4D mean motion model, 17 pts, TRE 3.3 +/- 1.6 mm); Liu et al. 2010 | Population motion priors adapted to a patient's lung shape | Lung only |
| Biomechanical lung registration | Al-Mayah 2009; Li 2013; Han 2017; Jafari 2021 | Patient-specific FE models reach about 1-2 mm; hybrid model + residual wins; sliding must be explicit | One organ, FE meshes built per patient |
| Supine/prone registration | Roth 2011 (DOI 10.1118/1.3577603), Hampshire 2013 (DOI 10.1016/j.media.2013.04.006) | Colon correspondence across posture change: 5.7-7.7 mm (fold-level) | Colon surface only |
| QA / uncertainty | AAPM TG-132 (Brock 2017); Meyer 2025; Smolders 2023 | No standard registration-uncertainty formalism; ensembles propagate into dose | Cross-cutting requirement |

**I did not find** (limited search: PubMed + 3 Consensus hits per query) a system that learns a patient's *joint model* (centres, axes,
limits) and soft-tissue coupling from their serial scans and uses it as the correspondence prior for voxel-level longitudinal
registration. The closest are Walter 2025 (hand-built articulated model) and Zhang 2026 (single scan). Do not claim novelty before a proper review.

## 3. Data  [verified via the TCIA catalogue API; sizes/licences as reported per series]

| Collection | Patients | Structure | Size | Licence | Use |
|---|---|---|---|---|---|
| **4D-Lung** | 20 | 82 4DCT + 507 4DCBCT (Hugo 2017, DOI 10.1002/mp.12059); 10 breathing phases; weekly 4DCT in 13 pts; daily 4DCBCT; implanted fiducials; external surrogate | 183 GB | CC BY 3.0 | **Primary**: breathing + day-to-day + weeks; fiducials are ground-truth landmarks |
| **CT COLONOGRAPHY** | 825 | supine + prone CT per patient (same session) | 497 GB | CC BY 3.0 | Posture/gravity change at scale: bones, ribcage, spine shape change |
| **Pelvic-Reference-Data** | 58 | 2 CTs per patient, median 11 days apart (max 40) | 7.2 GB | CC BY 3.0 | Small first pilot; organ filling variation |
| RIDER Lung CT | 32 | test-retest and repeat studies (some years apart) | 45 GB | CC BY 4.0 | Repositioning noise floor |
| Anti-PD-1_Lung | 46 | mostly 2 time points (median 57 d), CT + PET | 61 GB | CC BY 3.0 | Treatment-response change (non-motion) |
| QIN-SARCOMA | 15 | MR thigh/lower limb, 2-3 time points | 10 GB | CC BY 4.0 | Limb MR |
| Pancreatic-CT-CBCT-SEG | 40 | single date; CT + CBCT/RT structs | 14 GB | CC BY 4.0 | Abdomen CBCT |
| TotalSegmentator dataset v2.0.1 | 1228 | cross-sectional CT + 117-class masks (Zenodo 10047292) | 23.6 GB | CC BY 4.0 | Population anatomy prior (already used for the atlas) |

Not found in the public catalogue listing under these names: head-and-neck mid-treatment CT collections and NSCLC-Radiomics. The TCIA
"NLST" entry here is segmentation-heavy; full NLST imaging is a separate resource. Check IDC / NCI CDAS before relying on either.

Tooling facts [verified]: TCIA catalogue API and the IDC public S3 bucket are reachable from this sandbox; github.com releases are not,
but Zenodo hosts TotalSegmentator **v1** weights (e.g. record 6802342, CC BY 4.0), so a CPU pilot with automatic bone labels is possible.

## 4. Proposed model  [proposal]

Per-patient latent rig, population prior on top:

- **Static patient code `z_p`**: bone geometry, joint centres and axes, range-of-motion limits, soft-tissue coupling (which tissue
  follows which bone, sliding surfaces). Initialised from a population prior conditioned on the first scan.
- **Per-scan state `s_t`**: 6-DOF per bone (constrained by the joint model), breathing phase / depth, organ filling.
- **Decoder = the rig** (what this repo builds): rigid bones, pinned joints with limits, chain kinematics for the spine, skinned soft tissue,
  plus a small learned residual field, plus an explicit "no correspondence here" channel for non-motion change (shrinkage, resection, gas).
- **Learning signal**, all within one patient across scans: (a) per-bone rigid poses from segmentation; (b) functional joint centres: the
  point that moves least between two bones across poses (classical biomechanics, identifiable once a joint has moved between scans);
  (c) fiducials/landmarks; (d) image similarity; (e) cycle consistency over three or more scans; (f) Bayesian update of the
  population prior as scans accumulate (Rortveit-style), so uncertainty shrinks with every prior scan.
- **Uncertainty everywhere**: ensemble / posterior over `z_p`, carried into correspondence and dose.

Why "prior imaging teaches the rig" is plausible: each extra pose of a patient adds constraints on joint centres/axes/limits that
a single scan cannot give (a static scan shows a joint at one angle). That is testable as a **learning curve: error vs number of
prior scans**.

## 5. Staged experiments  [proposal]

Each stage has a pass/fail criterion and a baseline (uniGradICON / ANTs / VoxelMorph-style, from the literature above).

- **E0 synthetic (partly done):** the rig-deformed CT has exact voxel correspondence. Can the model recover rig parameters? Establishes identifiability and bounds.
- **E1 Pelvic-Reference-Data (7 GB):** pairs 11 days apart. Bone pose recovery; organ label propagation (Dice) vs baselines. Weak ground truth: no landmarks.
- **E2 4D-Lung:** fiducial TRE across phase, day and week; the learning curve vs number of prior scans; the headline test.
- **E3 CT colonography supine/prone:** bones/ribcage/spine across posture at scale; do patient-specific joints beat a population rig?
- **E4 generalisation:** leave-patient-out, leave-site-out; calibration of the uncertainty.
- Metrics: TRE (fiducials/landmarks), Dice of propagated labels, bone rigidity (Jacobian ~1 inside bones), folding fraction, inverse consistency, uncertainty calibration, effect on accumulated dose.

## 6. Risks and unknowns

- Joint centres from imaging are only good to a few mm; hard pins propagate that error, so constraints should be soft priors.
- Segmentation errors propagate (registration quality tracks segmentation quality); TotalSegmentator v1 lacks structures a full joint model needs (Walter 2025).
- CBCT quality/field-of-view truncation; MRI-only workflows (bone invisible).
- Anatomical variants (rib count, transitional vertebrae) mean the rig graph cannot be fixed.
- Non-diffeomorphic change (tumour response, surgery) breaks any pure-motion model.
- Ground truth for accumulated dose does not exist; fiducials and landmarks are the best proxies.
- Compute: only CPU here. Training needs GPUs.
- Licences: most are CC BY (attribution); MHR/SAM licences apply to the avatar side only, and not at all if the avatar is dropped.

## 7. Context needed from the project owner

1. Clinical focus first: thorax/lung (4D-Lung), head and neck, pelvis, or spine?
2. Which imaging do we have locally (planning CT, daily CBCT, MRI), and is institutional data usable (approvals)?
3. The target output: dose accumulation, contour propagation, response tracking, or patient positioning?
4. Compute available (GPU class, storage) and who maintains the data pipeline.
5. Is the camera-driven body model (the original repo goal) still a goal, or is it now only a demo for the rig?
