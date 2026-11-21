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

## The person, extracted and rebuilt

With the camera's pose known, "the person" is simply what lies in a box above the couch top in the room frame, after the
gantry-fixed pixels are removed. Ninety-one frames of the person sweep fuse into 4.9 million coloured points covering the
head and upper trunk from one side over the top to the other (`person.ply`, and `person.jpg` with front, side and top
views). The legs were never in view. The person breathes and the fusion is rigid, so the chest is blurred by that much.

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

The cheap thing to try first follows from the calibration: the gantry angle is known, so each picture can be turned to the
orientation the model likes best before it is given to it, and the fused cloud says where the body really is.

## Commands

    python tools/gantry_recording.py FOLDER info
    python tools/gantry_recording.py FOLDER sheet OUT.jpg
    python tools/gantry_recording.py FOLDER calibrate OUT_DIR
    python tools/gantry_recording.py FOLDER fuse OUT_DIR cube   --frames 300:1800:15  --box -200,200,60,460,-180,220 --flat-above -39
    python tools/gantry_recording.py FOLDER fuse OUT_DIR person --frames 4440:5530:12 --box -420,420,-500,1100,-70,380
    python tools/gantry_recording.py FOLDER pose OUT_DIR --frames 4410:5560:36      (needs the demo server running)

Boxes are x0,x1,y0,y1,z0,z1 in the room frame, millimetres: origin on the axis, Y along the axis, Z up.

## Not done yet

- **Absolute gantry 0 and the isocentre.** Angle 0 is "where the gantry stood at frame 0", and the origin is arbitrary
  along the axis. The treatment head is fixed in the camera's view, so the beam direction can be measured from its face and
  compared with the floor; that gives absolute gantry angle and, with the axis, the isocentre.
- **Calibration with the cube.** Detect the cube's markers, or register each frame's cube to the fused one, to remove the
  drift with angle and to tie the room frame to the cube.
- **Upload in the page** (depth, infrared, colour files instead of a live camera).
- **Pose given the angle**, and a head detector for extreme roll if that is not enough.
- The infrared stream is unused.
