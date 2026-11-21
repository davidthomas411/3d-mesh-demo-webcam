# Recordings from a camera on the gantry

Branch `gantry-recording`. A Femto Bolt rides on the gantry and records colour, depth and infrared while the gantry turns
round the couch: first with a calibration cube on the couch, then with a person. The aims, in order: get the person out of
the recording, rebuild a point cloud from all the views, see how the pose model copes with a camera that rolls with the
gantry, and tie every frame to the gantry's angle.

The recordings and everything made from them (pictures of a person) stay outside the repository, in `../recordings/gantry/`.

## The recording (2026-11-20_13-40-11)

| | |
|---|---|
| frames | 5772 over 288 s, 30 a second, none dropped in colour, depth or infrared |
| colour | 1280 x 720 (`color.mp4`) |
| depth | 16-bit millimetres, lossless, **already aligned to the colour picture and undistorted** (`depth.mkv`) |
| infrared | 16-bit, in the depth camera's own geometry (`ir.mkv`, not used yet) |
| gantry angle | **not recorded** |
| frames 0-2500 | the cube on the couch; one sweep of the gantry from frame 330 to 1700 |
| frames 2700-5772 | a person lying on the couch, head towards the gantry; one sweep from frame 4500 to 5520 |

The camera looks across at the treatment head, which therefore never moves in the picture, and the picture is rolled a
quarter turn: at the start position "down" in the room is to the right of the picture.

## Gantry angle and axis from the depth alone (`tools/gantry_geometry.py`)

No gantry angle was recorded, and none is needed for a first calibration. Because the camera rides on the gantry, the room
turns in the camera's frame about a fixed line, the gantry's axis. The room's large planes (floor, walls) are parallel to that
axis and at right angles to each other.

1. **Parts fixed to the gantry.** Pixels whose depth never changes over the recording (15% of the picture: the head).
   They are masked out of everything.
2. **Axis direction.** The three largest planes are fitted in each sampled frame; the axis is the direction their normals
   are all perpendicular to.
3. **Angle.** The direction of those normals about the axis gives the angle modulo 90 degrees. It is followed through time,
   taking the branch nearest the last angle and leaving out samples that would mean a jump (the room's planes hidden behind
   the person). Angle 0 is where the gantry stood at frame 0.
4. **Axis position.** The floor's distance from the camera at each angle fixes a point on the axis and the axis's height.

Result for this recording:

| | |
|---|---|
| sweeps | 0 to 171 degrees and back, twice (about 8 degrees a second) |
| axis height above the floor | 1242 mm |
| camera to axis | 778 mm |
| floor fit | 343 frames, 8 mm rms |

**Check with the cube.** One hundred frames of the cube sweep, each taken to the room frame with its own angle, fused into
one cloud (`fuse ... --flat-above`). If the geometry were wrong the cube would smear.

| Fused cube | |
|---|---|
| faces, thickness (rms) | 3.5 - 4.1 mm |
| angles between the three large faces | 88.2, 89.9, 86.4 degrees |
| edge length, from the faces' extent | about 120 mm |
| drift of a face with gantry angle | up to about 6 mm over the sweep |

So the depth-only calibration is good to roughly 4 mm. That is enough to reconstruct and to study pose by angle; it is not
enough for the clinical aim. The systematic drift with angle is what the cube is for (next steps).

## Calibration with the cube (`tools/aprilcube_target.py`, `tools/gantry_cube.py`)

The cube is an [aprilcube](https://github.com/younghyopark/aprilcube) target (MIT). No `config.json` came with the
recording, so the cube was identified from the pictures: 6x6 ArUco markers, ids 0-53, three by three on each face, which is
`aprilcube generate --grid 3x3x3 --dict 6x6_100 --tag-size 30` with the default one-cell margin and border, a 105 mm cube.
The border was confirmed by the reprojection error across two or more faces (0.86 px at one cell, 1.1-1.6 px at any other
width). aprilcube is not installed: its layout rules (face order, marker order, corner order) are restated in
`aprilcube_target.py`, which also reads an aprilcube `config.json` (`--config`). **The 30 mm marker size is inferred, not
known**: the depth puts the markers' spacing at 33.9 mm against 33.75 nominal. Measure the cube, or supply its config.

The cube stands still on the couch (on a stand, tilted, so that three faces show), so its pose in each picture is the
camera's pose in the cube's frame, and those poses lie on a circle about the gantry's axis.

1. **Pose per frame.** All visible markers at once (PnP, as aprilcube does). Then the position, not the orientation, is
   corrected with the depth on the cube's faces.
2. **Fit a rotation.** One axis, one point on it, one camera pose at angle 0, and an angle per frame.
3. **Table of departures.** What is left over (sag, wobble, and the pose's own noise), as medians in 10-degree bins, added
   back when a pose is asked for.
4. **Angle for frames without the cube.** The depth-only angle above, mapped by a straight line fitted on the cube sweep.

Two things turned up on the way and shaped the method:

- **Markers alone put the cube about 8 mm too near** (at 0.77 m, about 1%). The markers fix direction well and distance
  weakly, and either the cube is about 1% larger than assumed or the depth reads long (the Bolt's specification allows
  11 mm). The depth clouds are what is being fused, so the depth sets the distance.
- **The depth must not set the cube's orientation.** On faces this small and oblique the depth camera's planes are tilted
  by a few degrees (two faces of the cube measure 89.4 degrees apart in a single frame, 86.6 to 91.7 over the sweep).
  Letting them turn the cube put the gantry angle 5 degrees out at the far end of the sweep. The markers and the room's
  planes, which share nothing, agree: 170.4 and 171.0 degrees.

Result:

| | |
|---|---|
| frames with a cube pose | 475, of which 430 with two or more faces |
| sweep | -0.6 to 170.8 degrees |
| cube centre to the gantry axis | 4 mm (the cube was put at the isocentre, or very near) |
| camera to axis | 766 mm (depth only: 778) |
| depth-only calibration against the cube | axis direction 0.64 degrees off, axis position 10 mm off, angle 0.6 degrees rms (2.3 worst) |
| cube centre as the fitted rotation places it, over the sweep | 1.9 mm rms, 4.1 worst |
| the same with the table of departures | 0.7 mm rms, 2.0 worst (fitted and tested on the same sweep) |

**Check against the cube's true faces.** The depth of 167 frames, taken to the cube frame, against the six planes of a
105 mm cube (signed distance of every depth point lying over a face):

| Camera pose from | rms | 95% within | bias |
|---|---|---|---|
| depth-only calibration (above; thickness of the fused faces, not distance to truth) | 3.5 - 4.1 mm | | drift 6 mm |
| markers alone, each frame | 4.7 mm | 9.6 mm | 0.7 mm |
| markers + depth, each frame | 2.6 | 5.0 | 0.1 |
| fitted rotation at the cube's angle | 2.6 | 5.1 | 0.1 |
| **fitted rotation at the depth-only angle** (what a frame without the cube gets) | **2.7** | **5.2** | **0.1** |

So a frame of the person is placed as well as a frame in which the cube itself is visible. What is left, 2.6 mm, is no
longer the gantry model: a single depth frame is already 1.8 mm rms on a cube face and biased by up to 3 mm from face to
face. That is this depth camera on a small, oblique, black-and-white target; it is the floor for this check, and it is
not sub-millimetre.

Room frame from here on (`fuse ... --cube`): origin at the **cube's centre**, Y along the gantry axis, Z up. In the
depth-only frame the cube's centre is at (8, 271, 1) mm.

## The person, extracted and rebuilt

With the camera's pose known, "the person" is simply what lies in a box above the couch top in the room frame, after the
gantry-fixed pixels are removed. Ninety-one frames of the person sweep fuse into 4.9 million coloured points covering the
head and upper trunk from one side over the top to the other (`person.ply`, and `person.jpg` with front, side and top
views; `person_iso` is the same with the cube calibration, origin at the cube's centre). The legs were never in view. The person breathes and the fusion is rigid, so the chest is blurred by that much.

## Pose tracking from the rotating camera

The pose model was run on frames through the sweep, each given at four picture rotations (the demo server is used as the
model; `pose` command). Its mesh was then compared with the depth picture, which it was not shown: after one scale about
the camera, how far are the mesh's visible vertices from the measured surface?

| | |
|---|---|
| person found | every frame of the sweep, at all four rotations |
| 2-D | the mesh outline and keypoints follow the head, shoulders and arms at every gantry angle |
| 3-D, best rotation (270: body lying across the picture) | median 19-36 mm from the depth, 20-25 mm between 20 and 150 degrees |
| 3-D, worst rotation (180) | 33-61 mm |
| for comparison: seated, facing the camera, upright picture | about 7 mm |

So the model does not fail on a lying person seen from a rolling camera, but its 3-D pose is three to five times worse than
on an upright seated one, and which way up the picture is given matters by a factor of two. Lateral views (gantry near 0
and 170 degrees), where only the head and one shoulder are in the picture, are the worst. Whether a dedicated head detector
for extreme roll (DMHead was suggested) helps has not been tried; nothing has been downloaded.

### Judged in the room, with the calibrated camera (`tools/gantry_pose.py`)

The person lies still, so every frame's mesh, taken to the room frame with that frame's camera pose, should be the same
mesh in the same place. Thirty-two frames through the sweep (picture turned 270, one scale from the depth each), compared
with their own consensus over the 13,800 vertices the camera ever saw, and with the fused cloud:

| Gantry angle | orientation off | position off | left after a rigid fit | fused cloud to the mesh |
|---|---|---|---|---|
| 0 (from the side) | 6-8 deg | 82-112 mm | 15-22 mm | 42-61 mm |
| 20-35 | 4-6 | 32-57 | 14-22 | 16-25 |
| 45-115 (from above) | 2.4-5 | 13-30 | 14-19 | 14-22 |
| 125-150 | 4-8 | 38-70 | 13-19 | 18-24 |
| 165 (from the other side) | 10 | 91-98 | 24-25 | 36-39 |

(medians per frame; `pose_room_rot270.json`, picture `pose_room_rot270.jpg` with each frame's mesh coloured by angle.)

- **The pose does turn with the camera.** The body's orientation in the room stays within 2.4 to 10 degrees over a
  171-degree sweep, and the way out and the way back agree, so what is left is a function of the view, not noise.
- **Position is the large error, and it is along the line of sight.** The meshes fan out towards wherever the camera was:
  about 15 mm seen from above, about 100 mm seen from the side. One scale about the camera cannot fix this in a view where
  only the head and one shoulder show (docs/DEPTH_STUDY.md, experiment 1, on a real person).
- **Shape and articulation**: 13-25 mm is left after position and orientation are taken out, at every angle.
- **Knowing the gantry angle does not help the picture's orientation.** The camera turns about the body's long axis, so
  the body lies the same way up in the picture at every angle; only which side is seen changes. The four rotations above
  are the whole of that freedom. A head detector for extreme roll addresses a problem this geometry does not have. What the
  model struggles with here is a close (0.77 m), truncated, lateral view.

What the calibration does buy: the body's placement no longer has to come from each frame. One mesh fitted to the fused
cloud in the room, or the frames' meshes combined there, serves every angle, and its projection into any frame is known.

## Commands

    python tools/gantry_recording.py FOLDER info
    python tools/gantry_recording.py FOLDER sheet OUT.jpg
    python tools/gantry_recording.py FOLDER calibrate OUT_DIR
    python tools/gantry_recording.py FOLDER fuse OUT_DIR cube   --frames 300:1800:15  --box -200,200,60,460,-180,220 --flat-above -39
    python tools/gantry_recording.py FOLDER fuse OUT_DIR person --frames 4440:5530:12 --box -420,420,-500,1100,-70,380
    python tools/gantry_recording.py FOLDER pose OUT_DIR --frames 4410:5560:36      (needs the demo server running)
    python tools/gantry_pose.py FOLDER OUT_DIR --frames 4410:5560:36 --rot 270        (after cube and fuse person_iso; needs the demo server)
    python tools/gantry_recording.py FOLDER cube OUT_DIR --frames 0:2500:5          (after calibrate; --config for an aprilcube config.json)
    python tools/gantry_recording.py FOLDER fuse OUT_DIR person_iso --frames 4440:5530:12 --box -430,410,-770,830,-70,380 --cube

Boxes are x0,x1,y0,y1,z0,z1 in the room frame, millimetres: origin on the axis, Y along the axis, Z up.

## Not done yet

- **Absolute gantry 0.** Angle 0 is "where the gantry stood at frame 0". The treatment head is fixed in the camera's view,
  so the beam direction can be measured from its face and compared with the floor. The origin is now the cube's centre,
  4 mm from the axis; whether the cube was set to the lasers is not recorded.
- **The cube's true size**, and with it whether the depth reads 1% long. Callipers on the cube, or its `config.json`.
- **A depth error map.** The cube shows face-dependent depth errors of up to 3 mm. A larger flat target seen at several
  distances and angles would map them (docs/DEPTH_STUDY.md, experiment 3).
- **The table of departures is tested on the sweep it was fitted on.** A second cube sweep would test it properly, and
  would say how repeatable the gantry is.
- **Upload in the page** (depth, infrared, colour files instead of a live camera).
- **One body model for the whole sweep**: fit the mesh to the fused cloud in the room frame (the page's depth fit, with
  the cloud instead of one depth picture) and project it into each frame, instead of trusting each frame's placement.
- The infrared stream is unused.
