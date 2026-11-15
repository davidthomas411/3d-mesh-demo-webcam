"""Pose an atlas with a live MHR skin mesh (numpy reference implementation of anatomy.js).

Each structure is moved by the weighted rigid transform (Kabsch) that carries its supporting
skin vertices from the MHR rest pose to the live mesh. Bones, vessels and organs therefore
follow the body locally; there is no per-organ physics and no claim of internal accuracy.
"""
import json
import numpy as np


class Atlas:
    def __init__(self, prefix):
        self.meta = json.load(open(prefix + ".json"))
        blob = open(prefix + ".bin", "rb").read()
        self.structures = []
        for e in self.meta["structures"]:
            n, ns = e["n_verts"], e["support_n"]
            it = np.uint16 if e["idx_type"] == "u16" else np.uint32
            s = dict(e)
            s["pos"] = np.frombuffer(blob, np.float32, n * 3, e["pos"]).reshape(n, 3).astype(np.float64)
            s["faces"] = np.frombuffer(blob, it, e["n_tris"] * 3, e["idx"]).reshape(-1, 3).astype(np.int32)
            s["s_idx"] = np.frombuffer(blob, np.uint16, ns, e["support_idx"]).astype(np.int64)
            s["s_w"] = np.frombuffer(blob, np.float32, ns, e["support_w"]).astype(np.float64)
            s["s_rest"] = np.frombuffer(blob, np.float32, ns * 3, e["support_rest"]).reshape(ns, 3).astype(np.float64)
            self.structures.append(s)

    def names(self):
        return [s["name"] for s in self.structures]

    def pose(self, skin_posed):
        """skin_posed: (18439,3) live MHR vertices in the SAME frame/units as the rest mesh (metres)."""
        out = {}
        for s in self.structures:
            R, c_rest, c_live = weighted_kabsch(s["s_rest"], skin_posed[s["s_idx"]], s["s_w"])
            out[s["name"]] = (s["pos"] - c_rest) @ R.T + c_live
        return out


def weighted_kabsch(src, dst, w):
    w = w / w.sum()
    cs, cd = w @ src, w @ dst
    H = ((src - cs) * w[:, None]).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cs, cd
