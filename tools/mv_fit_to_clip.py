"""Convert the multi-view SAM 3D Body fit of the multi-view-breathing-mesh project into a clip file for this rig (tools/mhr_clip.py).

Run in that project's SAM3D environment, from its `mv/` folder (it imports marker_fit for the MHR head), after mv_fit.py has been run with the
parameter export described in docs/MOCAP_CLIPS.md (mv_fit.npz must contain `body` (T,133), plus the fixed `global_rot`, `hand`, `scale`, `shape`, `expr`).
    python mv_fit_to_clip.py tmp/mv_fit/mv_fit.npz OUT.npz --video-fps 60
Each frame's MHR model parameters are produced with the head's own mhr_forward(return_model_params=True): [global_trans*10, global_rot, body pose,
hands replaced, scales] = 204 values, the same field SAM 3D Body returns. NOT run against real data in the sandbox that wrote it.
"""
import sys
import numpy as np
import torch


def main():
    a = sys.argv[1:]
    src, out = a[0], a[1]
    vfps = float(a[a.index("--video-fps") + 1]) if "--video-fps" in a else 60.0
    z = np.load(src, allow_pickle=True)
    for k in ("body", "global_rot", "hand", "scale", "shape", "expr", "frames"):
        if k not in z.files:
            raise SystemExit(f"{src} has no '{k}': apply the parameter export to mv_fit.py first (docs/MOCAP_CLIPS.md)")
    import marker_fit as MF                                   # the project's own module (SAM3D env, run from mv/)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = MF.Fitter().head
    t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev).reshape(1, -1)
    frames = np.asarray(z["frames"]); body = np.asarray(z["body"]).reshape(len(frames), -1)
    P = []
    with torch.no_grad():
        for b in body:
            r = head.mhr_forward(global_trans=torch.zeros(1, 3, device=dev), global_rot=t(z["global_rot"]), body_pose_params=t(b),
                                 hand_pose_params=t(z["hand"]), scale_params=t(z["scale"]), shape_params=t(z["shape"]),
                                 expr_params=t(z["expr"]), return_model_params=True)
            P.append(r[-1][0].detach().cpu().numpy())         # last element is model_params when only that flag is set
    P = np.stack(P).astype(np.float32)
    step = float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0
    np.savez(out, mhr_model_params=P, shape_params=np.asarray(z["shape"]).reshape(1, -1), frames=frames, fps=vfps / step,
             source=f"multi-view SAM 3D Body fit ({src})")
    print("wrote", out, P.shape, "fps", vfps / step)


if __name__ == "__main__":
    main()
