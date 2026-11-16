# Driving the skeleton and CT with a motion-capture clip

Status: the pipeline runs end to end on real open motion capture (Microsoft Rocketbox, below) and on a SYNTHETIC test clip. No SAM 3D Body clip has been run yet.

Why no real clip: the SAM 3D Body checkpoints (`facebook/sam-3d-body-dinov3`, `facebook/sam-3d-body-vith`) are gated on Hugging Face (anonymous download returns 401;
manual access request). No public per-frame MHR parameter sequences from SAM 3D Body were found. Unofficial re-uploads of the weights exist but were not used:
they bypass the access request and a `.ckpt` is a pickle, so loading an unverified one executes arbitrary code.

Ways to get a real clip:
1. Run `tools/sam3d_video_to_clip.py VIDEO clip.npz` in a sam-3d-body checkout with approved access (GPU advised). It is untested against the real model.
2. Give this sandbox a Hugging Face token with approved access (as an environment secret) and a short video; frames can be run on CPU (slow).
3. Any source of per-frame `mhr_model_params` (204 values per frame, SAM 3D Body's own field) saved as `.npz` (`mhr_model_params`, optional `fps`, `source`) or `.json`.

Pieces: `tools/mhr_clip.py` (loader; smoothing; global translation/rotation and per-person scale neutralised because the patient lies in the CT frame and the atlas is the
mean-shape MHR), `experiments/render_clip.py` (AP + lateral bone radiographs, coronal + sagittal real-HU reformats for the best model), `clips/make_synthetic_clip.py`
(24-frame reach-and-turn with neck flexion and an arm raise, to exercise the path). Memory: `warped_volume` now works in slabs, one render worker is safe.

## Reusing the multi-view capture from the linac / SGRT project (found in the repos, not in a chat)

Prior chats were not readable from here (only three code sessions are visible; none was the linac digital twin). Reviewing the repositories instead:
- `davidthomas411/multi-view-breathing-mesh`: six calibrated GoPro views of a supine volunteer, SAM 3D Body in every camera, a markerless multi-view MHR fit
  (`mv/mv_fit.py`: one MHR body optimised so its vertices project onto every camera's SAM 3D Body vertices; median 41 mm to Vicon markers vs 114 mm for a typical
  single camera), Sapiens2 keypoints, ~1 mm skin tracking. It already uses SAM 3D Body weights, so access exists on that machine (Windows, RTX 4060, 8 GB).
  The study data and videos are not in the repository, so they are not available here.
- `tju-thomas-lab/treatment-room-simulation` (DepthCam Sim): the linac room, camera presets and Blender scene; a place to put the rig-driven patient later.

What is reusable for this rig: the multi-view fit's per-frame MHR body is exactly what the clip format needs, but `mv_fit.py` currently saves only world-frame vertices (`X`),
not parameters. Two small edits to `mv_fit.py` (not applied; the repository was cloned read-only):
1. after `out = dict(frames=[], X=[], loss=[], resid=[], dropped=[])` add `out["body"] = []`;
2. in the frame loop, next to `out["X"].append(...)`, add `out["body"].append(body.detach().cpu().numpy().reshape(-1))`, and in both `np.savez(...)` calls add
   `body=np.array(out["body"]), global_rot=p["global_rot"].cpu().numpy(), hand=p["hand"].cpu().numpy(), scale=p["scale"].cpu().numpy(), shape=p["shape"].cpu().numpy(), expr=p["expr"].cpu().numpy()`.
Then, in that project's SAM3D env from its `mv/` folder: `python <this repo>/tools/mv_fit_to_clip.py tmp/mv_fit/mv_fit.npz clip.npz --video-fps 60`, and
`python experiments/render_clip.py clip.npz OUTDIR` here. `mv_fit_to_clip.py` is untested against real data.
Caveats: global translation/rotation and the per-person scale are discarded (the patient lies in the CT frame; the atlas is mean-shape); the fit's frame stride is 6, so a 60 fps video
gives about 10 fps motion.

## Open-source mocap that works now: Microsoft Rocketbox motions (MIT)

`huggingface.co/datasets/codepawl/rocketbox-motions-glb` (ungated, MIT, 326 clips, 30 fps, 121-joint Bip01 skeleton, `skeleton/<clip>/<clip>.glb` is the bare skeleton).
`tools/rocketbox_to_clip.py CLIP.glb OUT.npz --mhr-onnx mhr-body.onnx` retargets by fitting the 62 non-hand MHR body parameters so 15 bone/body-axis directions
(trunk, hip line, shoulder line, clavicles, upper arms, forearms, thighs, shins, head-up, eye line) match the Biped's forward kinematics (scale-free). Mean direction error
0.3 deg on `f-gestic-shrug-01`; parameters max 0.57 rad, smooth. Limits: pelvis motion, root translation, fingers and face are not transferred; spine twist/bend is only
constrained through shoulder, hip and neck keypoints. Clips with a shrug, look-around, stretch-arms and breathing exist (`f-gestic-shrug-01`, `f-idle-look-around-01`,
`f-idle-stretch-arms-01`, `f-idle-breathe-01`); no bending or walking clips were found in the list. scipy's `least_squares` stalled at poor minima here (float32 ONNX,
forward differences); the tool uses its own Levenberg-Marquardt with central differences.
Output: `clips/rocketbox_shrug.npz`; render with `experiments/render_clip.py`.

CT-warp fixes this needed: the brain now takes exactly the skull's rigid motion in the CT frame (separate fits differed by ~3 deg / 15 mm and left a lattice-shaped hole pattern
in the skull), and competing bone owners at a joint are tried in distance order.

Look-around clip (`f-idle-look-around-01`): retargeted to `clips/rocketbox_look_around.npz` (46 frames at 10 fps, mean direction error 0.9 deg; the pelvis line is the worst direction at 5.8 deg because
pelvis motion is not transferred). Rendered through the rig and CT; the skull and brain stay intact through a head turn of well over 90 degrees. Remaining defect: the face and jaw soft tissue tears
under large head rotation (the smooth soft-tissue field is stretched across the head-neck shear zone); not fixed.

Skull hole: the couch/arm removal mask was cutting 2945 skull voxels (1.4%) at the occiput, a hole in the skull at every pose; the mask now never removes skeleton or brain voxels.

Original motion next to the rig: `tools/rocketbox_original_panel.py AVATAR_ENGINE.glb CLIP_FRAMES_DIR OUT_DIR` skins the Rocketbox avatar (`engine/<clip>/<clip>.glb`) with its own animation and draws the Bip01
skeleton over it in front and side views, pasted left of the rig/CT frames (same frame times, same axes). The first front/side figures (`docs/rig_*.png`) and the first version of this panel drew the
front view mirrored (person-left on the image left); fixed so a viewer facing the person sees person-left on the right, matching the radiograph panels.

Gallery: `site/` is a static page (`index.html`, `data/motions.json`, `media/*.mp4|jpg`) showing eight Rocketbox motions with the original beside the rig + CT render, built by
`experiments/make_gallery.py`. It can be served from any static host (for GitHub Pages: publish `site/`).
