# Patient-specific, drivable biomechanical rig learned from serial imaging

Status: design + data survey (no model trained yet). Everything marked **[verified]** was retrieved in the session that wrote
this (TCIA catalogue API, PubMed, Consensus). Everything marked **[proposal]** is a design decision, not a result.

## 1. The problem

Images are snapshots and registration is blind to anatomy. We want a model that gives **voxel-to-voxel correspondence across
time points** (planning CT, daily CBCT, mid-treatment, follow-up) by representing each patient as a *drivable rig*: rigid bones,
joints with patient-specific centres/axes/limits, and soft tissue that moves with them. The rig is learned **per patient** from
that patient's own prior scans, on top of a population prior.

## 1b. Decisions from the project owner (this session)

- **Site:** lung first (open data, 4D-Lung); head and neck is the clinically more interesting follow-on.
- **Data:** open-source only for training and testing, for now.
- **First domain output:** patient positioning. **Stage 1 (funded): camera-driven body model for computer-vision SGRT**, treated as an early goal.
- **High-level goal:** image-quality improvement through a patient prior (reconstruction from fewer projections / lower dose).
- **Compute:** A6000 plus a second 24 GB-class GPU when needed (exact second card to be confirmed). CPU-only in this sandbox.

## 1c. Roadmap

| Stage | Output | Needs | Status |
|---|---|---|---|
| 1 | Camera -> body mesh (SAM 3D Body + Femto Bolt depth) -> rig -> 6-DOF positioning and surface-to-anatomy | depth fusion, ROI 6-DOF, rig | depth fusion + rig prototype built; joints/respiration/JS port pending |
| 2 | Per-patient rig learned from serial scans, lung (4D-Lung) | segmentation, per-bone pose, joint model, residual | plan only |
| 3 | Head and neck | open H&N longitudinal data (not yet located) | data search pending |
| 4 | Reconstruction with a patient prior (sparse-view / low-dose CBCT) | simulated projections (raw projections are not in TCIA), GPU | design only |

## 1d. Key experiments for the owner's priorities  [proposal]

**S1: does a rig beat rigid surface registration at locating internal anatomy? (stage 1, CPU-feasible pilot).**
From a 4D-Lung CBCT take the body surface (a stand-in for the camera surface) and the true bone/fiducial positions. Compare, per scan:
(a) rigid-ROI surface registration (what SGRT does), versus (b) rig-based inference of bone poses and tumour/fiducial position from the surface plus the planning CT.
Metric: residual bone/fiducial/tumour error (mm, degrees). Context for the baseline [verified]: lung SGRT vs CBCT residuals are roughly 1.5-1.9 mm SD and 0.7-0.8 deg (Guo 2023, J Appl Clin Med Phys), with
SGRT mostly improving translations and not rotations (Gosztonyi 2026). Constraint: the CBCT is a 450 x 450 x 150 mm slab (0.88 x 0.88 x 3 mm, head-first supine), so the rig must work on partial field of view.

**S2: respiration.** Drive the rig's rib/diaphragm/lung model with breathing phase and test on the 10 phases of the planning 4DCT and on the implanted fiducials.
Prior art on this exact dataset [verified]: Huang 2022 (Med Phys) predicts internal deformation from surface motion (DIR-Lab landmarks < 2 mm; 4D-Lung tumour Dice 0.835).

**S3: reconstruction with a rig-warped prior.** Simulate sparse-view/low-dose CBCT from CTs (4D-Lung ships reconstructed images only, not raw projections), then compare FDK vs prior-constrained reconstruction with the prior being
(a) the undeformed planning CT, (b) a generic deformed prior, (c) the rig-warped prior. Prior art [verified]: adaptive PICCS using an earlier CBCT as the prior allows a claimed 10-40x dose reduction (Lee 2012);
deforming the prior by projection-based registration improved PICCS (XCAT, 20 projections: RMSE 14% -> 10%; Zhang 2014). A rig supplies a far lower-dimensional deformation, which should make projection-based pose estimation well posed from very few views.

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
| Surface to internal anatomy | Huang et al. 2022, *Med Phys* (surface motion -> deformation PCs; XCAT, DIR-Lab, 4D-Lung) | Internal deformation predicted from body surface; DIR-Lab landmarks < 2 mm; 4D-Lung tumour Dice 0.835 | Breathing only; no skeleton/joint structure; no positioning output |
| Surface to synthetic anatomy | Douglass et al. 2023, *Phys Eng Sci Med* (pix2pix, optical scan -> cranial MRI) | Much gross cranial anatomy is predictable from external shape; limited where population variation is wide | Image-to-image, not a drivable rig; no ground truth for a real patient |
| Prior-image reconstruction | Lee et al. 2012 (APICCS), Zhang et al. 2014 (deformed prior, DPIR), *Phys Med Biol* / *Med Phys* | Earlier images as priors cut dose 10-40x; deforming the prior before PICCS helps | Generic deformation, not skeleton-aware; estimated per scan |
| SGRT clinical accuracy (lung) | Guo 2023; Shim 2026; Gosztonyi 2026 | SGRT residual vs CBCT about 1.5-1.9 mm SD / 0.7-0.8 deg; improves translation more than rotation | Baseline to beat |
| QA / uncertainty | AAPM TG-132 (Brock 2017); Meyer 2025; Smolders 2023 | No standard registration-uncertainty formalism; ensembles propagate into dose | Cross-cutting requirement |

**I did not find** (limited search: PubMed + 3 Consensus hits per query) a system that learns a patient's *joint model* (centres, axes,
limits) and soft-tissue coupling from their serial scans and uses it as the correspondence prior for voxel-level longitudinal
registration. The closest are Walter 2025 (hand-built articulated model) and Zhang 2026 (single scan). Do not claim novelty before a proper review.

## 3. Data  [verified via the TCIA catalogue API; sizes/licences as reported per series]

| Collection | Patients | Structure | Size | Licence | Use |
|---|---|---|---|---|---|
| **4D-Lung** | 20 | 82 4DCT + 507 4DCBCT (Hugo 2017, DOI 10.1002/mp.12059); 10 breathing phases; weekly 4DCT in 13 pts; daily 4DCBCT; implanted fiducials; external surrogate. Catalogue: 5,070 CBCT phase volumes (Varian Trilogy, 50 slices, 450 x 450 x 150 mm, ~26 MB each) + 820 planning/weekly 4DCT phase volumes (~110 slices); ~27 imaging dates per patient. Images only (no raw projections) | 183 GB (~4 MB/s transfer, about 1 min per patient-day) | CC BY 3.0 | **Primary**: breathing + day-to-day + weeks; fiducials are ground-truth landmarks |
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
- Compute: only CPU here. Training needs the A6000-class GPUs. Raw CBCT projections are not public here, so reconstruction work uses simulated projections.
- Partial field of view: the CBCT slab is 15 cm craniocaudal; the rig must work with a handful of vertebrae and ribs.
- Head and neck: the public TCIA listing did not show longitudinal H&N CT/CBCT collections under the names searched; locate one (e.g. via IDC) before committing.
- Licences: most are CC BY (attribution); MHR/SAM licences apply to the avatar side only, and not at all if the avatar is dropped.

## 7. Open questions

1. Confirm the second GPU (24 GB class) and available storage for ~200 GB of DICOM.
2. Head and neck: is there any open longitudinal H&N CT/CBCT you already know of? Otherwise this stays a later, data-limited stage.
3. For SGRT: which camera/surface system (AlignRT-like, Femto Bolt, other) and which positioning tolerances (mm/deg) define success?
4. Does stage 1 need a real-time requirement (frame rate/latency) that constrains how heavy the rig can be?

## 8. Performance: measured (4-core Xeon CPU, no GPU) and what real time would take

Measured in this sandbox (single process, quiet machine):

| Stage | Time | Note |
|---|---|---|
| MHR body model (ONNX, CPU) | 0.02 s | |
| Rig: pose all 114 structures (numpy) | 0.04 s | not a bottleneck |
| Depth anchor | 8 ms | |
| Projective ICP (4 iterations, 18k points, numpy) | 129 ms | |
| Software z-buffer (pure-Python triangle loop, 37k triangles) | 0.65 s, called twice per frame | the real problem in `DepthFusion`: 1.30 s per frame in total |
| CT warp: displacement field (KD-tree, 32 neighbours) | 5.1 s | CPU nearest-neighbour search |
| CT warp: source positions (field upsample + bone override, 15 M voxels) | 5.4 s | single-thread SciPy |
| CT warp: resample CT and bone mask (15 M voxels) | 3.4 s | |
| Projections (AP + lateral) | 0.16 s | |
| Whole radiograph/reformat frame | 15.9 s | one-off setup 68 s; 140 frames took about 30 min with 3 workers on 4 cores |

Cited, not measured here: Fast SAM 3D Body about 75-100 ms body-only and 140-170 ms end to end on an RTX A6000 (this repo's README).

Not measured (estimates for a GPU port; verify before relying on them): the warp, resample and projection steps are embarrassingly parallel and
should fit in tens of milliseconds on a 24 GB-class GPU (the CT is ~116 MB); the displacement field would have to move from per-frame
nearest-neighbour search to precomputed skinning weights, which changes the model and needs re-validation; the z-buffer and ICP would move to a GPU renderer/torch.

Real-time design for stage 1: the mesh model sets the pace (about 6-13 Hz); between mesh updates a depth-only fast path (anchor + ICP against the
reference surface) can run at camera rate. The CT-warp radiograph is a visualisation and a prior-image generator, not part of the positioning control loop.
