# Driving the skeleton and CT with a motion-capture clip

Status: the pipeline runs end to end, but only on a SYNTHETIC test clip. No real SAM 3D Body clip has been run yet.

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
