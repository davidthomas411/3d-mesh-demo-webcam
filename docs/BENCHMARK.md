# Anatomical-realism benchmark

One real CT (TotalSegmentator dataset subject s1159, CC BY 4.0), one fixed body motion (the MHR parameter script in
`tools/render_radiology.py`), five stages of the model. Every stage is deformed from the same CT, so differences in the
pictures and the scores come only from the model. New realism features are added as new columns; the input never changes.

Run: `python experiments/benchmark_versions.py metrics out.json` (scores, no CT needed) and
`python experiments/benchmark_versions.py render outdir` (side-by-side frames; needs the CT, the correspondence file from
`tools/ct_correspondence.py` and the MHR ONNX; about 12 min on 4 CPU cores).

## Stages

| Stage | What it adds | Mode |
|---|---|---|
| V0 skin-driven warp | nothing anatomical: the body surface drives all tissue, bones smear | `structures=False` in `tools/ct_warp.py` |
| V1 per-structure fits | every structure chases its own skin patch (the first rig) | `PerStructure` in `experiments/benchmark_versions.py` |
| V2 rigid segments + spine chain | coherent skeleton: segments from skin, vertebrae as a kinematic chain, ribs on vertebrae | `atlas.pose(skin, joints=False)` |
| V3 + joint rules | pinned ball/saddle/condyloid joints, clavicle aim at the shoulder cap, scapulohumeral rhythm, range-of-motion limits, rigid brain | `atlas.pose(skin)` |
| V4 + in-vivo spine split | trunk rotation shared along the vertebrae by measured per-level data | `atlas.pose(skin, spine="physio")` (numpy only; browser port pending) |

Planned, not implemented: breathing (ribs, diaphragm, lungs), scapulothoracic contact, soft-tissue sliding, costal-cartilage flexibility,
coupled bending-rotation, natural motion timing, asymmetry.

## Scores (shoulder shrug peak, twist peak, forward-bend peak)

| Stage | Humeral head vs scapula under shrug (mm change) | Lumbar share of twist (in-vivo reference 28%) | Correlation with in-vivo per-level rotation profile | Max vertebra spacing change (mm) |
|---|---|---|---|---|
| V1 per-structure fits | 1.0 | 49% | -0.14 | 4.3 |
| V2 rigid chain, no joints | **12.2** (ball sinks into the scapula) | 35% | -0.53 | 1.5 |
| V3 + joint rules | 1.9 | 35% | -0.53 | 1.5 |
| V4 + in-vivo spine split | 1.9 | **26%** | +1.00 | 1.6 |

How to read the scores honestly:

- **The V4 profile correlation is a conformance score, not independent validation:** V4 uses the same per-level table as the reference.
  The independent evidence is that V1-V3, which do not use it, correlate poorly or negatively.
- **V1's shoulder score is good by coincidence** (humerus and scapula both fit nearby skin). It fails on the spine.
- A min-surface-gap score for joints was tried and dropped: overlapping bones also have a gap near zero, so it missed the V2 dislocation.
  The head-centre-to-scapula distance does not. Gap changes at the sternoclavicular and acromioclavicular joints (V3/V4 3.7-4.2 mm, V2 0.8 mm) are not
  a quality score: V2 looks better only because it does not move the clavicle at all.
- The V0 column has no scores: it has no skeleton to score.

## Evidence behind V4's per-level weights

| Quantity | Source | Status |
|---|---|---|
| Axial rotation per thoracic level T1-T2..T12-L1 (1.2, 1.6, 1.4, 1.6, 1.8, 1.9, 2.3, 2.5, 2.7, 2.6, 1.3, 0.5 deg to one side) | Fujimori et al. 2012, *Spine*, in vivo 3D CT, 13 volunteers, neutral vs maximum trunk rotation | measured |
| Lumbar axial rotation 0.6-2.2 deg per level, largest L1-L2..L3-L4; about 2 deg per level, L3-4 and L4-5 slightly more | Ochia et al. 2006, *Spine* (CT, 15 volunteers); Pearcy et al. 1984, *Spine* (3D radiography) | measured ranges; the five values used (0.8, 1.6, 2.0, 2.0, 1.8) are chosen inside them: **assumed** |
| Regional axial rotation (thoracic ~35, pelvis ~30, lumbar ~8 deg) | Narimani et al. 2018, *J Biomech* (22 adults, inertial sensors) | measured |
| Lateral bending and flexion splits | Narimani 2018 (regional totals), Pearcy 1984 (lumbar lateral bending 10:10:10:6:3), Pan et al. 2018 review (lower thoracic contributes more) | regional totals measured; per-level splits **assumed** (uniform within region except as cited) |
| Coupling of bending and rotation | Pearcy 1984, Narimani 2018, Orbach 2023: present but inconsistent between subjects | **not modelled** |

Total modelled pelvis-to-T1 axial rotation is whatever the skin segments report; the table only decides how it is shared among the vertebrae.
Per-level literature values differ between methods (Grankova et al. 2026 show the split depends strongly on the kinematic model), so treat the
distribution as a defensible default to be fitted to imaging data, not as ground truth.

## Known limitations of every stage

No breathing, no scapulothracic contact, no soft-tissue sliding, arms and couch removed from the CT, and the CT is one supine subject, so
internal anatomy here is that subject's, warped by the model, not a patient's own. The deformation of tissue between bones is an interpolation
of the modelled structures' motion (Gaussian weighting of landmark motion, rigid inside bones), not a physical simulation.

## Premise audit: no gaps between bones, no bones out of joints (`experiments/joint_audit.py`)

72 bone pairs touch or nearly touch at rest. For each, over three motions x seven poses, the worst change from rest: gap opened (mm) and
interpenetration added (mm); a pair counts as a violation above 3 mm. Interpenetration is exact (signed distance to the watertight
mesh) except where the other bone's mesh is open (scapula, cartilage), where it falls back to a nearest-vertex sign test that over-reports.
Raw results: `docs/joint_audit_results.json`.

**Correction:** the first version of this audit used the nearest-vertex test for every pair and reported vertebra-to-vertebra overlaps of up to
16 mm and a clavicle-cartilage overlap of 18 mm. With the exact test the vertebral overlaps are at most about 1 mm, and the exact
reverse check of the clavicle against the cartilage found no overlap. Those findings were artefacts.

| Stage | Pairs opened > 3 mm | Pairs overlapping > 3 mm | Main offenders |
|---|---|---|---|
| V1 per-structure fits | 17 | 7 | everything; no coherent skeleton |
| V2 rigid chain, no joints | 1 | 3 | humeral head sinks into the scapula |
| V3 + joint rules | 1 | 2 | clavicle-sternum opens 2.6-3.8 mm; scapula/humerus/clavicle contacts (open mesh, indicative) |
| V4 + in-vivo spine split | 1 | 3 | same as V3; T11/T12 opens 2.5 mm |

The real defect found: costal cartilage was one rigid body carried by the ribs, so in V3/V4 the rib-cartilage junctions opened by up to 9 mm
(V4: 10 pairs opened before the fix). Cartilage is flexible, so it is now a link whose ends follow the nearest rib and the sternum
(`Atlas._cartilage`, ported to `anatomy.js`, parity-tested); this closed the rib-cartilage and cartilage-sternum junctions in the audit.

Still open: the sternoclavicular joint opens up to 3.8 mm (the saddle joint pins a centre, not the articular surfaces), scapulothoracic contact is
not modelled, and there is no collision resolution. The route to a guarantee by construction is joint coordinates as the unknowns plus a
penetration-resolving step; this audit is the regression test for it.

## Broken neck (atlas v7): groups re-joined by a joint solve (`tools/close_junctions.py`)

`atlas_build` placed each bone group (head_neck, thorax, lumbar, pelvis) with its own rigid fit to the avatar's skin, so the spine was disconnected at the
group boundaries at rest, which the change-from-rest audit could not see. Gap between the junction vertebrae at rest (CT truth / atlas v6 -> v7):

| Junction | CT subject | v6 | v7 |
|---|---|---|---|
| sacrum / L5 | 1.0 mm | 1.7 | 0.7 |
| L1 / T12 | 1.1 mm | 23.2 | 1.4 |
| T1 / C7 | 0.6 mm | 9.2 (centroids 22 mm too far apart) | 1.1 |

One 18-parameter solve (a rigid correction per group) trades junction offset, junction kink, skin containment of the core bones and size of the
move; the core bones stay inside the avatar skin (head_neck 4.9% -> 5.6% of sampled vertices, thorax and lumbar 0%). Corrections are large (thorax moved
60 mm, rotated 17 deg; head and neck 48 mm, 12 deg) because the avatar's torso proportions differ from the CT subject's; the discrepancy is now carried
by the group placements instead of by gaps in the spine. The CT correspondence file is updated by the same transforms. Picture: `docs/compare_neck_fix.png`
(first model V0 | V4 before | V4 now, four frames). The pinned-joint test tolerance for ball joints was relaxed from 1.5 to 2.5 mm (the right glenohumeral
change under extreme motion is 2.0 mm on v7); that is within in-vivo glenohumeral translation but is a loosening, not a free pass.
