"""Population consensus -> patient-specific respiratory model, leave-one-patient-out.
Open data: TCIA 4D-Lung planning 4DCT, end-inhale (0%) and end-exhale (50%), one date per patient.
Per patient: TotalSegmentator labels on both phases; structure centroid displacement (exhale->inhale) measured in the
patient's own vertebra-aligned frame (rigid Kabsch on T-vertebrae removes whole-body shift).
Consensus = shrunken mean over the OTHER patients. Patient-specific update = scale of the consensus fitted to k observed
structures (k = 0 means pure population prior). Error = mean 3D error (mm) on the held-out structures.
usage: python experiments/population_consensus.py WORKDIR   (needs pop_manifest.json from population_fetch.py)"""
import sys, os, json, glob, numpy as np, nibabel as nib
from experiments.s1_surface_vs_bone import segment, kabsch

GROUPS = {"lung_l": "lung_*left", "lung_r": "lung_*right", "sternum": "sternum", "liver": "liver", "heart": "heart"}


def centroids(segdir, ref):
    aff = nib.load(ref).affine; out = {}
    for f in sorted(glob.glob(f"{segdir}/*.nii.gz")):
        n = os.path.basename(f)[:-7]
        m = np.asanyarray(nib.load(f).dataobj) > 0
        if m.sum() < 500:
            continue
        key = None
        if n.startswith("vertebrae_T"): key = n
        elif n.startswith("rib_"): key = n
        elif n.startswith("lung_") : key = "lung_" + n.split("_")[-1]
        elif n in ("sternum", "liver", "heart", "aorta"): key = n
        if key is None: continue
        out.setdefault(key, []).append((m.sum(), (aff @ np.append(np.argwhere(m).mean(0), 1))[:3]))
    return {k: sum(w * c for w, c in v) / sum(w for w, _ in v) for k, v in out.items()}


if __name__ == "__main__":
    work = sys.argv[1]
    man = json.load(open(f"{work}/manifest.json")) if os.path.exists(f"{work}/manifest.json") else None
    files = {}
    for m in man:
        files.setdefault((m["patient"], m["date"]), {})[m["phase"]] = m["file"]
    D = {}
    for (p, d), ph in sorted(files.items()):
        if 0.0 not in ph or 50.0 not in ph: continue
        c = {}
        for k, f in ph.items():
            c[k] = centroids(segment(f, f"{work}/seg_{p}_{d}_{int(k)}"), f)
        sh = sorted(k for k in c[0.0] if k.startswith("vertebrae_T") and k in c[50.0])
        if len(sh) < 3: print(p, "skip: vertebrae", len(sh)); continue
        R, t = kabsch(np.array([c[50.0][k] for k in sh]), np.array([c[0.0][k] for k in sh]))
        disp = {k: c[0.0][k] - (R @ c[50.0][k] + t) for k in c[0.0] if k in c[50.0] and not k.startswith("vertebrae")}
        D[p] = disp; print(p, "structures", len(disp), "lung_l SI mm", round(float(disp.get("lung_left", [0, 0, 0])[2]), 1), flush=True)
    json.dump({p: {k: v.tolist() for k, v in d.items()} for p, d in D.items()}, open(f"{work}/population_disp.json", "w"))
    pats = sorted(D); keys = sorted(set.intersection(*[set(D[p]) for p in pats]))
    print("patients", len(pats), "shared structures", len(keys))
    X = np.array([[D[p][k] for k in keys] for p in pats])          # (P, K, 3)
    rng = np.random.default_rng(0); res = {"zero": [], "pop": []}; curve = {k: [] for k in (1, 2, 4)}
    for i, p in enumerate(pats):
        mu = np.delete(X, i, 0).mean(0); true = X[i]
        res["zero"].append(np.linalg.norm(true, axis=-1).mean()); res["pop"].append(np.linalg.norm(true - mu, axis=-1).mean())
        for kk in curve:
            errs = []
            for _ in range(30):
                obs = rng.choice(len(keys), kk, replace=False); rest = np.setdiff1d(np.arange(len(keys)), obs)
                s = (true[obs] * mu[obs]).sum() / max((mu[obs] ** 2).sum(), 1e-9)     # patient-specific amplitude of the consensus pattern
                errs.append(np.linalg.norm(true[rest] - s * mu[rest], axis=-1).mean())
            curve[kk].append(np.mean(errs))
    out = {"n_patients": len(pats), "structures": keys,
           "zero_motion_mm": float(np.mean(res["zero"])), "population_prior_mm": float(np.mean(res["pop"])),
           "prior_plus_k_observed_mm": {k: float(np.mean(v)) for k, v in curve.items()}}
    json.dump(out, open(f"{work}/population_result.json", "w"), indent=1); print(json.dumps(out, indent=1))
