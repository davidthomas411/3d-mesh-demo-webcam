"""Turn a short video into a clip file of per-frame MHR parameters with SAM 3D Body. RUN WHERE THE MODEL IS AVAILABLE (needs the
SAM 3D Body code, its dependencies and the gated checkpoints from Hugging Face; a GPU is strongly advised). Not run in the sandbox
that wrote it: the checkpoints are gated and were not accessible there, so this script is untested against the real model.

usage (from a sam-3d-body checkout):  python sam3d_video_to_clip.py VIDEO OUT.npz [--stride N] [--max-frames N] [--hf-repo facebook/sam-3d-body-dinov3]
Writes mhr_model_params (T, 204) for the highest-confidence person per frame, plus fps and shape_params.
"""
import sys
import numpy as np
import cv2


def main():
    a = sys.argv[1:]
    video, out = a[0], a[1]
    stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 1
    maxf = int(a[a.index("--max-frames") + 1]) if "--max-frames" in a else 10 ** 9
    repo = a[a.index("--hf-repo") + 1] if "--hf-repo" in a else "facebook/sam-3d-body-dinov3"
    from notebook.utils import setup_sam_3d_body
    est = setup_sam_3d_body(hf_repo_id=repo)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    P, S, i = [], [], 0
    while len(P) < maxf:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            outs = est.process_one_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if outs:
                o = outs[0]
                P.append(np.asarray(o["mhr_model_params"]).reshape(-1)); S.append(np.asarray(o["shape_params"]).reshape(-1))
                print("frame", i, "ok", flush=True)
            else:
                print("frame", i, "no person; repeating previous", flush=True)
                if P: P.append(P[-1]); S.append(S[-1])
        i += 1
    np.savez(out, mhr_model_params=np.stack(P), shape_params=np.stack(S), fps=fps / stride, source=f"SAM 3D Body ({repo}) on {video}")
    print("wrote", out, len(P), "frames")


if __name__ == "__main__":
    main()
