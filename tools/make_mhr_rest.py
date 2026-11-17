"""Write the neutral MHR body the atlas builder needs: rest vertices, faces and the rest KEYPOINTS (the 70 body keypoints SAM 3D
Body reports, which anatomy/atlas_build.py and anatomy/rig.py index as shoulders 5/6, elbows 7/8, hips 9/10, knees 11/12, neck 69).
The keypoints are a fixed linear map of the mesh vertices and skeleton joints (head.keypoint_mapping in the checkpoint).

usage: python tools/make_mhr_rest.py mhr_model.pt model.ckpt OUT.npz [faces_from.npz]"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bind_mhr_joints import MHR  # noqa: E402


def main():
    import torch
    mhr_path, ckpt_path, out = sys.argv[1:4]
    v, jp, _ = MHR(mhr_path).pose()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    key = [k for k in sd if k.endswith("keypoint_mapping") and sd[k].shape[1] == len(v) + len(jp)]
    if not key:
        raise SystemExit("no keypoint_mapping of the right width in the checkpoint")
    M = sd[key[0]].numpy().astype(np.float64)
    kp = M @ np.vstack([v, jp])
    faces = np.load(sys.argv[4])["faces"] if len(sys.argv) > 4 else None
    np.savez_compressed(out, verts=v, joints=kp, **({"faces": faces} if faces is not None else {}))
    print(f"{key[0]} {M.shape}: wrote {out} (verts {v.shape}, keypoints {kp.shape}); hips y {kp[9][1]:.4f} {kp[10][1]:.4f}, neck y {kp[69][1]:.4f}")


if __name__ == "__main__":
    main()
