"""Fetch planning 4D-CT end-inhale (0%) and end-exhale (50%) volumes for several 4D-Lung patients (TCIA, CC BY 3.0).
usage: python experiments/population_fetch.py WORKDIR [--patients 8] [--dates 1]
Network only; segmentation is a separate step (TotalSegmentator, CPU)."""
import sys, os, json, urllib.request, urllib.parse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from s1_surface_vs_bone import get, fetch_nifti

if __name__ == "__main__":
    work = sys.argv[1]
    n_pat = int(sys.argv[sys.argv.index("--patients") + 1]) if "--patients" in sys.argv else 8
    n_dates = int(sys.argv[sys.argv.index("--dates") + 1]) if "--dates" in sys.argv else 1
    os.makedirs(work, exist_ok=True)
    sr = json.loads(get("getSeries", Collection="4D-Lung"))
    big = [s for s in sr if s["Modality"] == "CT" and s["ImageCount"] > 60]
    per = {}
    for s in big:
        per.setdefault(s["PatientID"], {}).setdefault(s["StudyDate"][:10], []).append(s)
    import re
    phase_of = lambda s: (float(m.group(1)) if (m := re.search(r"(\d+(?:\.\d+)?)%", s["SeriesDescription"])) else None)
    # keep patients whose first 4DCT date has every phase labelled (some series carry no phase label: skipped, not guessed)
    ok = [p for p in sorted(per) if all(phase_of(s) is not None for s in per[p][sorted(per[p])[0]])]
    print("patients with labelled phases:", len(ok), "of", len(per), flush=True)
    pats = [ok[i] for i in np.unique(np.linspace(0, len(ok) - 1, n_pat).round().astype(int))]
    manifest = []
    for p in pats:
        for d in sorted(per[p])[:n_dates]:
            for ph in (0.0, 50.0):
                m = [s for s in per[p][d] if phase_of(s) == ph]
                if not m:
                    continue
                out = f"{work}/{p}_{d}_{int(ph)}.nii.gz"
                fetch_nifti(m[0], out)
                manifest.append({"patient": p, "date": d, "phase": ph, "file": out})
                print("fetched", os.path.basename(out), flush=True)
    json.dump(manifest, open(f"{work}/manifest.json", "w"), indent=1)
    print("done", len(manifest), "volumes")
