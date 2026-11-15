"""Joint integrity on REAL MHR articulations (needs mhr-body.onnx; not run in CI).

usage: PYTHONPATH=. python tests/validate_joints.py ATLAS_PREFIX mhr-body.onnx
For every joint: gap = smallest distance between the child's surface near the joint centre and the parent's
surface, at rest and under each pose. A pinned joint keeps its gap (socket stays a socket); a bone that
slides off its parent shows as a change of centimetres.
"""
import sys
import numpy as np
import onnxruntime as ort
from scipy.spatial import cKDTree
from anatomy.runtime import Atlas

atlas = Atlas(sys.argv[1])
sess = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
def skin(params):
    p = np.zeros((1, 204), np.float32)
    for i, v in params.items():
        p[0, i] = v
    return sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)

by = {s["name"]: s for s in atlas.structures}
joints = [j for j in atlas.meta["joints"] if j["type"] != "costovertebral"]
PARENT_NAME = {"hip_l": "hip_left", "hip_r": "hip_right"}

def gap(j, posed):
    """min distance child(near J) <-> parent, using the REST selection of child vertices near the joint."""
    J = np.asarray(j["center"]); c = by[j["child"]]["pos"]
    sel = np.linalg.norm(c - J, axis=1) < 0.035
    if sel.sum() < 5:
        sel = np.argsort(np.linalg.norm(c - J, axis=1))[:30]
    parent = j["parent"]
    return cKDTree(posed[parent]).query(posed[j["child"]][sel])[0].min()

rest = {n: s["pos"] for n, s in by.items()}
rest_gap = {j["name"]: gap(j, rest) for j in joints}
poses = {
    "shrug both":        {31: 0.3, 41: 0.3},
    "shrug + protract":  {31: 0.3, 41: 0.3, 32: 0.25, 42: 0.25},
    "arm raise (shoulder joint)": {30: 0.4, 34: 0.4, 40: -0.4, 44: 0.4},
    "head turn + nod":   {25: 0.4, 26: 0.3},
    "spine mix":         {8: 0.3, 10: -0.3, 14: 0.2, 19: 0.3},
    "all":               {8: 0.25, 16: 0.25, 31: 0.3, 41: 0.3, 25: 0.3, 30: 0.3, 34: 0.3},
}
print("rest gap (mm): " + ", ".join(f"{k} {1000 * v:.1f}" for k, v in rest_gap.items()))
print(f"\n{'pose':28s} " + " ".join(f"{j['name']:>8s}" for j in joints) + "   <- change of gap vs rest, mm")
for name, prm in poses.items():
    posed = atlas.pose(skin(prm))
    print(f"{name:28s} " + " ".join(f"{1000 * (gap(j, posed) - rest_gap[j['name']]):8.1f}" for j in joints))
