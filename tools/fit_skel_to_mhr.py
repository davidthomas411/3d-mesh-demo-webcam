"""Fit SKEL (Keller et al. 2023: a biomechanical skeleton inside an SMPL-topology skin, with bone sizes and positions learned from
data) to an MHR body, so there is a learned reference for where bones sit inside that body.

SKEL's skin has SMPL topology; Meta's MHR repository publishes a barycentric map from the MHR mesh to SMPL vertices. The MHR mesh
is carried into SMPL topology with that map and SKEL's shape, pose and translation are optimised so its skin lies on it. No SMPL
model file is needed. The result (SKEL bones and joints in the MHR/atlas frame) is written to an .npz.

SKEL is under a non-commercial research licence: keep the model files and anything exported from them out of the repository.

usage: python tools/fit_skel_to_mhr.py mhr_rest.npz mhr2smpl_mapping.npz skel_models_v1.1 out.npz [--gender male|female|both]"""
import argparse, os, sys
import numpy as np
import torch

FLIP = np.array([1.0, -1.0, -1.0])           # atlas frame (y down, front -z) <-> SKEL frame (y up, front +z)


def mhr_to_smpl_topology(verts, faces, mapping):
    tri = faces[mapping["triangle_ids"]]
    return (verts[tri] * mapping["baryc_coords"][:, :, None]).sum(1)


def fit(gender, model_dir, target, iters=(250, 500), device="cpu"):
    from skel.skel_model import SKEL
    model = SKEL(gender=gender, model_path=model_dir).to(device).eval()
    tgt = torch.as_tensor(target, dtype=torch.float32, device=device)[None]
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    poses = torch.zeros(1, 46, device=device, requires_grad=True)
    trans = torch.zeros(1, 3, device=device, requires_grad=True)
    with torch.no_grad():
        out = model(poses, betas, trans, skelmesh=False)
        trans += (tgt.mean(1) - out.skin_verts.mean(1))
    # stage 1: placement and shape; stage 2: pose as well
    for stage, n in enumerate(iters):
        params = [trans, betas] + ([poses] if stage else [])
        opt = torch.optim.Adam(params, lr=0.02 if stage == 0 else 0.01)
        for _ in range(n):
            opt.zero_grad()
            out = model(poses, betas, trans, skelmesh=False)
            v2v = (out.skin_verts - tgt).norm(dim=-1)
            loss = (v2v ** 2).mean() + 1e-4 * (betas ** 2).mean() + 1e-4 * (poses ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = model(poses, betas, trans, skelmesh=True)
        v2v = (out.skin_verts - tgt).norm(dim=-1)[0].cpu().numpy() * 1000
    return model, out, v2v, betas.detach().cpu().numpy()[0], poses.detach().cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rest"); ap.add_argument("mapping"); ap.add_argument("skel_dir"); ap.add_argument("out")
    ap.add_argument("--gender", default="both")
    a = ap.parse_args()
    rest = np.load(a.rest)
    verts, faces = np.asarray(rest["verts"], np.float64), np.asarray(rest["faces"]).reshape(-1, 3)
    target = mhr_to_smpl_topology(verts, faces, np.load(a.mapping)) * FLIP
    print(f"MHR mesh in SMPL topology: {target.shape[0]} vertices, height {np.ptp(target[:, 1]):.3f} m")
    best = None
    for g in (("male", "female") if a.gender == "both" else (a.gender,)):
        model, out, v2v, betas, poses = fit(g, a.skel_dir, target)
        print(f"  SKEL {g:6s}: skin-to-skin distance mean {v2v.mean():.1f} mm, median {np.median(v2v):.1f} mm, 95% {np.percentile(v2v, 95):.1f} mm")
        if best is None or v2v.mean() < best[3].mean():
            best = (g, model, out, v2v, betas, poses)
    g, model, out, v2v, betas, poses = best
    skel_v = out.skel_verts[0].cpu().numpy() * FLIP
    import pickle
    bones = pickle.load(open(os.path.join(a.skel_dir, f"skel_{g}.pkl"), "rb"))     # which skeleton vertices belong to which bone
    np.savez_compressed(
        a.out, gender=g, betas=betas, poses=poses, v2v_mm=v2v,
        skin_verts=out.skin_verts[0].cpu().numpy() * FLIP, skin_faces=model.skin_f.cpu().numpy(),
        skel_verts=skel_v.astype(np.float32), skel_faces=model.skel_f.cpu().numpy().astype(np.int32),
        joints=out.joints[0].cpu().numpy() * FLIP, joints_ori=out.joints_ori[0].cpu().numpy(),
        bone_names=np.array(list(model.bone_names)), bone_verts_indices=np.asarray(bones["bone_verts_indices"]),
        joints_name=np.array(list(bones["joints_name"])))
    print(f"kept SKEL {g}; wrote {a.out} ({len(skel_v)} skeleton vertices)")


if __name__ == "__main__":
    main()
