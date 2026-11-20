# Kinematic tree: bones connected by construction

Premise: a drivable skeleton with no gaps between bones and no bone out of its joint, whatever the input motion. Status as measured by
`experiments/joint_audit.py`, `experiments/spine_limits_audit.py` and the tests; history in `docs/PROGRESS.md`.

## Target design
- Nodes are rigid bones; edges are joints with a centre, type and limits (the atlas joint table already has centre/type/limit for gh, sc, ac, hip, ao, costovertebral).
- Unknowns are joint coordinates (angles about the centre, small translations where the joint allows them), not per-bone poses.
- Input (camera skin, later imaging) enters as targets for a few bones or segments; a solve finds the joint coordinates that fit them subject to limits.
- A penetration-resolving step projects interpenetrating pairs apart along the joint's free directions.
- Soft tissue deforms between bones (skinning now; physics/sliding later).

## What is already true by construction
| Property | Mechanism | Evidence |
|---|---|---|
| Ball joints stay seated (gh, hip) | child rotates about the joint centre, centre pinned to the parent's image | pin test, <= 2.5 mm joint-space change under extreme motion |
| Spine is a connected chain | forward kinematics over centroids, spacing preserved; atlas v7 re-joined the group boundaries (0.7-1.4 mm vs CT 0.6-1.1 mm) | audit, `docs/compare_neck_fix.png` |
| Rib-cartilage-sternum loop stays closed | cartilage is a flexible link whose ends follow the nearest rib and the sternum | audit: rib-cartilage openings from 9 mm to < 3 mm |
| Scapula stays on the thorax | contact step about the AC pin (atlas v8) | forward flexion gap 12-15 mm (was up to 27) |
| Range of motion per vertebral level | not enforced; audit shows benchmark motions are far inside assumed limits (max 3.6 deg per level) | `spine_limits_audit.py` |

## Not yet true (the remaining tree work)
1. Joint coordinates are not the solved unknowns: segment poses still come from skin first, joints are imposed afterwards, so extreme inputs can ask for impossible poses and rely on clamps.
2. No collision/penetration step: arm abduction pushes the scapula 2-3 mm into the ribs; sternoclavicular joint opens up to ~2.4 mm (saddle joint pins a centre, not surfaces).
3. Intervertebral limits are not enforced and facet/spinous-process contact is not modelled; the spine range table is assumed, not fitted.
4. Group placements in the atlas carry a large offset from the CT subject (thorax 60 mm) because the avatar proportions differ; a per-patient build should solve placement and
   joint coordinates together.
5. The browser port lacks the physio spine split and the no-joints stage.

## Next steps, in order
1. Joint-coordinate solve for the spine and rib cage (Gauss-Newton on angles with limits, targets from the skin segments), reproducing the current poses where they are valid.
2. Penetration projection for the shoulder girdle (scapula/ribs/clavicle) using exact meshes where watertight; fix the open scapula/cartilage meshes first.
3. Fit per-level weights and limits from imaging (open 4D-Lung / CT Colonography supine-prone pairs) instead of assumed tables.
