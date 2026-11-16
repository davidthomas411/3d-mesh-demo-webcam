# Real-time rig with Fast SAM 3D Body

Goal: drive the rig live from the Fast SAM 3D Body demo (`webcam_server.py` + `webcam.html`, WebSocket `/stream`, one 18,439 x 3 float32 mesh per camera frame).

## Where the time goes
| Stage | Time | Source |
|---|---|---|
| Fast SAM 3D Body, body mode | 75-100 ms (RTX A6000), ~65 ms end to end (RTX 5090) | this repo's README, Fast SAM 3D Body paper; not measured here (no GPU, gated weights) |
| Mesh over the WebSocket | 221 KB per frame | protocol |
| Browser rig: segment fits, physiologic spine, joints, scapular contact, all 114 structures | 3.5-4.4 ms mean (node, 4-core Xeon, no other load) | `tests/js/bench_rig.mjs` |
| Same rig inside the live loop with the replay server on the same 4 cores | 5.7 ms mean, 12.9 ms p95 | `tests/js/live_client.mjs` |
| Live radiograph (WebGL, 123 bone meshes drawn twice) | 44-57 ms in software GL here; a GPU does this in well under a millisecond | `tests/xray_shot.mjs` |
| CT-warp reformats and radiographs (offline renderer) | about 15 s per frame on CPU | not real time |

So the rig itself is not the bottleneck: it costs about 4 ms against an estimator that delivers a mesh every 75-100 ms, 100-170 ms after capture.

## What was done
- **Physiologic spine in JS** (`spine="physio"`, the best model, V4): ported from numpy, parity tested (`tests/test_joints.py`). `AnatomyLayer` uses it by default.
- **Display-rate pose stream** (`anatomy/stream.js`): keeps the last two meshes with their capture times and extrapolates the vertex velocity to the present, capped at 120 ms, so the rig can run every display frame
  (60 Hz) from a 10-13 Hz estimator and shed part of its latency. Simulated on real retargeted motion (`tests/js/stream_sim.mjs`), mean vertex error against the person's true pose at display time:

  | Motion, estimator | Hold last mesh | Extrapolate (120 ms cap) |
  |---|---|---|
  | Shake arms (fast), 15 Hz, 90 ms latency | 9.7 mm | 5.5 mm |
  | Shake arms, 12 Hz, 120 ms | 12.3 mm | 8.2 mm |
  | Shake arms, 10 Hz, 170 ms | 16.6 mm | 12.8 mm |
  | Look around, 12 Hz, 120 ms | 6.3 mm | 2.5 mm |

  Smoothing on top added lag and made it worse, so it is off by default. This models latency and update rate only; estimator noise (jitter) is separate and the demo's existing filters still apply.
- **Live radiograph** (`anatomy/xray.js`): AP and lateral radiographs of the posed bones in WebGL, drawn every frame from the rig's own geometries (front faces add -depth, back faces +depth, so each pixel accumulates path length through bone).
  Open meshes (scapula, costal cartilage) add a fixed thin layer. Off by default in the page ("Live X-ray"). Checked in headless Chromium (software GL) with `tests/xray_shot.mjs`; not yet seen on a real GPU or with the real page.
- **Demo wiring** (`webcam.html`): "Predict between updates" (on by default) and "Live X-ray" in the anatomy panel; `webcam_server.py` and the replay server serve `stream.js` and `xray.js`.
- **Test without a GPU** (`tools/replay_server.py`): same `/stream` protocol, answers each camera frame with the next mesh of a recorded MHR clip after a simulated inference delay.
  `node tests/js/live_client.mjs ws://localhost:8000/stream atlas/atlas 10` measures mesh rate, capture-to-arrival latency and rig cost.
  With a 90 ms simulated estimator on a loaded 4-core CPU: 7.8 Hz meshes, 127 ms capture-to-arrival (p95 146), rig 5.7 ms mean, 56 fps display.

## Not done / needs the GPU machine
- Run `webcam_server.py` with the real Fast SAM 3D Body and the page in a browser; none of the page wiring has been exercised in a real browser (esm.sh three.js was not reachable here).
- The offline rig numbers above are CPU-only; no measurement of GPU frame time for the radiograph.
- Extrapolation is on vertices, which is crude for fast rotations; extrapolating segment transforms (slerp) would be better. Camera-translation (`pred_cam_t`) is not extrapolated.
- A CT reformat in real time needs a GPU port of the warp with precomputed skinning weights (a model change, see docs/PATIENT_RIG_PLAN.md section 8); not started.
- Clinical use: none. Illustrative only.
