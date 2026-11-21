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

## The linac's own gantry angle (`tools/gantry_console.py`)

A GoPro filmed the linac's console through the session (`GX010105.MP4`, 3840 x 2160, 302 s, upside down). The page shows
"Gantry Angle ... Actual" to a tenth of a degree. The number is read in every frame: the field is cropped, its characters
separated, and each matched against ten digit shapes clustered from the video itself (labelled once, by eye).

| | |
|---|---|
| frames read | 8992 of 9044 (the rest are caught mid-repaint) |
| linac angle over the session | 359.7 to 170.5 degrees, i.e. -0.3 to 170.5 |
| console repaints while moving | about 4 a second, so the reading is a staircase; values are taken at the moment they appear and joined by straight lines |
| time offset, console video minus recording | 9.3 s, found from the two angle traces |
| **linac angle at the recording's frame 0** | **359.9 (-0.1): "angle 0" above is gantry 0** |

Against the linac:

| Angle from | scale | zero | rms, gantry still | rms, moving | worst |
|---|---|---|---|---|---|
| the cube's markers (cube session) | 0.9996 | -0.07 deg | 0.12 deg | 0.40 | 1.0 |
| depth only, cube session | 1.004 | -0.56 | | 0.64 overall | |
| depth only, person session | | | | 2.8 overall | 12.6 |

- The cube calibration's angle is the linac's to about a tenth of a degree at rest. While moving, the 0.4 degrees is mostly
  the console's four repaints a second (the gantry turns 2 degrees between repaints), not the cube.
- The depth-only angle is about a degree out through the person's sweep and jumps by up to 12.6 degrees when the person
  moves in front of the room's planes (around frame 5680, gantry still). It was only ever checked on the cube session.
  **Where a console video exists the tools now use the linac angle** (`gantry_cube.frame_angles`); the depth-only angle
  remains the fallback.
- The console also shows the couch position, which changed between the cube and the person (one readout 65.3 to 74.9 cm,
  another -9.6 to -12.6 cm). The room frame is fixed to the room, not the couch, so this does not move the calibration,
  but it is the record of where the couch was and could be read the same way.

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

## Tracking motion while the gantry turns (`tools/gantry_track.py`)

The practical use is to follow a patient who may move while the gantry rotates, so nothing may be fused over time during
tracking: every depth frame has to give a measurement on its own. The method is the one docs/DEPTH_STUDY.md recommends,
moved into the room frame:

1. A frame's depth is taken to the room frame with the camera pose for its gantry angle (the cube calibration).
2. A region is a box in the room frame (head: 220 x 245 mm; chest: 460 x 360 mm).
3. The region's live points are registered rigidly, point-to-plane, onto a reference surface. The result is the region's
   six shifts for that frame.

The reference needs the patient still once, for one sweep, as any surface-guidance reference capture does. Two kinds were
compared: **matched** (the region as seen from the same gantry angle in the reference sweep) and **fused** (one surface
from the whole reference sweep).

**Test.** The person's outward pass (frames 4500-5004) is the reference; 42 frames of the return pass are "live", each
measured alone. Nobody measured what the person really did, so this is an upper bound on the method's error.

| Region, reference | apparent shift, median | 95% | first 3 s after the turn-round |
|---|---|---|---|
| chest, matched | 1.5 mm | 2.4 | 0.8 - 1.4 (one frame 3.2) |
| chest, fused | 1.4 | 2.2 | 0.9 - 1.9 |
| head, matched | 4.4 | 7.1 | 0.1 - 1.9 |
| head, fused | 3.3 | 4.9 | 0.8 - 1.5 |

(gantry angle from the linac; `track_linac.json`. With the depth-only angle the medians are the same within 0.2 mm and
the chest's 95% is 3.4 mm instead of 2.4; `--angles planes`.)

- **The head figure is mostly a real movement, found by the tracking.** Just after the turn-round, where reference and
  live are a second or two apart, the head reads 0.1-0.8 mm. From about frame 5120 it drifts steadily to 5-7 mm sideways
  (room X) over the next 12 s, the same in both references and with either source of gantry angle, while the chest
  stays at 1-1.5 mm. An angle or timing error would move the chest too. So the person's head rolled about 5 mm between
  the passes; it cannot be proved without an independent measurement.
- **The floor is about 1 mm** (0.5-1.5), read from the frames nearest the turn-round and from the chest. It is not
  sub-millimetre. The chest includes breathing.
- **The gantry angle was not the limit after all.** Replacing the depth-only angle (about 1 degree out in this sweep)
  with the linac's removed the worst frames and left the medians where they were. What is left is the depth camera on
  one frame (1.8 mm rms on a flat cube face), breathing, and the console's coarse timing while moving.
- Matched and fused references perform alike here.
- A known motion applied to the live points (2, -3, 1.5 mm and about 1 degree) is recovered to 0.1 mm against the
  matched reference, but that test re-uses the frame's own points, so it shows the registration converges, not how
  accurate it is. Against the fused reference the same test is off by 2.6 mm for the head and 0.4 mm for the chest:
  a head-sized region slides on a 4 mm fused surface. Regions need to be large and the fused surface finer.

### In the live page: mesh, point cloud, skeleton, CT (`tools/gantry_replay.py`)

The live page runs on the recording as if it were the camera. `gantry_replay.py` writes the colour frames as a video
(turned head-up) and the depth of the same frames as the server's recorded-depth format; `run_gantry_replay.cmd` starts
the server on them (port 8096); then

    http://127.0.0.1:8096/?bolt=1&video=/gantry-data/replay.mp4

starts by itself: the pose model on every picture, the mesh fitted to the depth, the point cloud (tick "Depth point
cloud" in the 3-D view's settings), the skeleton, the DRR and the CT, all as live. The recording's depth is already
aligned to the colour picture, so the replay's "depth camera" is the colour camera. The page sends the video's time with
every picture and the server serves the depth frame of that moment, so the two stay in step (`RecordedCloud.seek`).

On this recording the depth fit reports the skin a median 0.7 mm from the depth, 87% within 2 mm, at about 30 fits a
second.

**Room frame.** The camera rides on the gantry, so in its own frame the whole scene turns. With `--geo OUT_DIR`,
`gantry_replay.py` also writes the camera's pose for every frame (cube calibration, linac angle), and the page then shows
the 3-D view in the room frame: the patient lies still on the couch, origin at the cube's centre, and the treatment head
and the picture's viewpoint go round. Mesh, skeleton and point cloud are all still computed in the camera's frame; the
scene as a whole is moved, frame by frame. `&room=0` in the address shows the camera's frame again. The picture panel,
the DRR and the CT are views from the camera or of the body itself and are unchanged.

What is on screen is not all from one moment (the mesh comes from an older picture than the point cloud), and at
8 degrees a second that shows, so each is placed with the camera pose of its own moment. The server says which moment of
the recording every depth frame it serves belongs to, and the page says where its video is each time it asks for one;
without both the point cloud turned with the gantry whenever the page fell behind. A yellow pyramid marks where the
camera is now and which way it looks, and a faint line its path round the patient over the whole recording.

**The depth fit on a moving camera.** Run with a picture and a depth frame of the same instant, the fit succeeds at every
gantry angle (skin a median 0.5 mm from the depth). Live it kept dropping out: the mesh comes from a picture taken a
quarter of a second or more before the depth frame it is fitted to, and the camera has turned several degrees by then.
The server now carries each mesh into the depth frame's viewpoint with the known camera motion before fitting
(`RecordedCloud.camera_motion`, `tracker.moved_model`). Through the whole loop it then holds about 30 fits a second with
the skin 0.6-0.8 mm from the depth (2 mm at moments).

**Not live, not shown.** With the depth fit on, the body and skeleton are drawn only while the depth camera is placing
them: if nothing has been fitted for 0.6 s, or the fit is only holding its last placement, they are hidden and the shift
bars freeze, instead of falling back to the model's own mesh (a guess at distance).

**DRR in the room.** The DRR's viewpoint and centre are fixed in the room for a replay (from above the couch, head up), so
a still patient gives a still radiograph while the gantry turns and the green reference shows the patient's own movement.
The CT slice stays with the body while a reference is held (a camera-fixed plane would turn with the gantry).

**Head-up display** (bottom left of the 3-D view, H hides it): gantry angle and its source, frame, fit rate, skin-to-depth
distance, latencies, how far the depth is behind the video, what is drawn, the reference and its shifts, and flags in
red: no fits, holding, not live, out of tolerance. It also says the shifts come from the body mesh.

**Only where there is data.** The model always returns a whole body, also the legs the camera never saw. "Body and
skeleton only where the depth camera has data" (3-D view settings; on by default for a replay, available live too) keeps
a fragment of the mesh or skeleton only if the depth camera measured a surface along its line of sight that the fragment
lies on or up to 45 cm behind. It is done in the shaders with the depth camera's own lens model, so it follows every
depth frame. The reference ghost is left whole.

**Reference tracking in the room frame.** With a replay that has camera poses, "Capture reference" and the six bars work
in room coordinates: live mesh and reference are each taken to the room with the camera pose of their own moment, LAT is
across the couch, LNG along it, VRT up-down, and the reference ghost stays where it was captured while the gantry moves
on. The position is the centre of the selected region's vertices as placed by the depth fit, not the model's own distance
estimate (which made the bars swing by 400 mm). Through the sweep, on a person lying still, the bars read mostly 2-30 mm
(40 mm at worst) and under 1 degree once the depth fit holds: that is the body mesh placed by the depth, not the depth
registration's 1 mm, which is not wired into the bars yet.

### The tracking analysis as a player

`http://127.0.0.1:8097/gantry` plays an exported recording: the colour picture (head up) with the tracked regions tinted,
slices across head and chest with the frame's depth over the reference surface, the six shifts of each region for the
frame on screen, and charts of the shifts and the gantry angle over time (click or drag in a chart to go to that moment;
space plays, arrow keys step). `tools/gantry_export.py` makes the export, one frame at a time against the surface fused
from the reference sweep. The export is pictures of a person, so it is written outside the repository and the server
reads it from `GANTRY_WEB` (default `../recordings/gantry/web`). This is a player for a finished analysis; uploading
depth, infrared and colour files through the page is still not built.

What this means for the tool: the pose model names the regions and follows gross motion (its orientation holds to
2.4-10 degrees through the sweep); the numbers come from registering each frame's depth in the room frame; the reference
is one still sweep; the gantry angle comes from the machine (here, read off its console; a direct feed would also give
the timing the console's repaint rate hides).

## Commands

    python tools/gantry_recording.py FOLDER info
    python tools/gantry_recording.py FOLDER sheet OUT.jpg
    python tools/gantry_recording.py FOLDER calibrate OUT_DIR
    python tools/gantry_recording.py FOLDER fuse OUT_DIR cube   --frames 300:1800:15  --box -200,200,60,460,-180,220 --flat-above -39
    python tools/gantry_recording.py FOLDER fuse OUT_DIR person --frames 4440:5530:12 --box -420,420,-500,1100,-70,380
    python tools/gantry_recording.py FOLDER pose OUT_DIR --frames 4410:5560:36      (needs the demo server running)
    python tools/gantry_pose.py FOLDER OUT_DIR --frames 4410:5560:36 --rot 270        (after cube and fuse person_iso; needs the demo server)
    python tools/gantry_track.py FOLDER OUT_DIR --ref 4500:5004:3 --live 5010:5508:12  (reference pass, live pass; --angles planes|linac)
    python tools/gantry_replay.py FOLDER ../recordings/gantry/web --frames 4380:5580:2 --geo OUT_DIR ; run_gantry_replay.cmd   (the live page on the recording)
    python tools/gantry_export.py FOLDER OUT_DIR ../recordings/gantry/web --frames 4380:5580:3   (then open /gantry in the web app)
    python tools/gantry_console.py read VIDEO CONSOLE_DIR ; ... digits CONSOLE_DIR ; ... angle CONSOLE_DIR
    python tools/gantry_console.py sync CONSOLE_DIR FOLDER OUT_DIR                     (linac angle per frame of the recording)
    python tools/gantry_recording.py FOLDER cube OUT_DIR --frames 0:2500:5          (after calibrate; --config for an aprilcube config.json)
    python tools/gantry_recording.py FOLDER fuse OUT_DIR person_iso --frames 4440:5530:12 --box -430,410,-770,830,-70,380 --cube

Boxes are x0,x1,y0,y1,z0,z1 in the room frame, millimetres: origin on the axis, Y along the axis, Z up.

## Not done yet

- **The isocentre along the axis.** Gantry 0 is now known from the console. The origin is the cube's centre, 4 mm from the
  axis; whether the cube was set to the lasers is not recorded.
- **The cube's true size**, and with it whether the depth reads 1% long. Callipers on the cube, or its `config.json`.
- **A depth error map.** The cube shows face-dependent depth errors of up to 3 mm. A larger flat target seen at several
  distances and angles would map them (docs/DEPTH_STUDY.md, experiment 3).
- **The table of departures is tested on the sweep it was fitted on.** A second cube sweep would test it properly, and
  would say how repeatable the gantry is.
- **Upload in the page** (depth, infrared, colour files instead of a live camera).
- **A recording with a known motion** (a phantom moved by a measured amount) to give the tracking a true error, and a
  gantry angle with proper timing (a direct feed, or the console filmed with a shared clock).
- **Averaging over a few frames and a projective registration**, the two things docs/DEPTH_STUDY.md says take a single
  frame's 1 mm towards a few tenths.
- **Regions from the pose model** instead of fixed boxes in the room.
- The infrared stream is unused.
