"""Pictures of the underlying rig: skeleton by articulated group, every joint (type-coded), the spine chain, the scapular contact patch,
flexible cartilage links, at rest and in a pose; plus a kinematic-tree diagram.
usage: ATLAS=.. REST=.. ONNX=.. python experiments/rig_figure.py OUTDIR [clip.npz frame]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anatomy.runtime import Atlas
from anatomy import rig

GROUP_COL = {"head_neck": "#e8c07a", "thorax": "#9fb8d8", "lumbar": "#e29a86", "pelvis": "#b79ad8", "arm": "#8fcf9f", "leg": "#d8d39a", None: "#cccccc"}
JOINT_COL = {"ball": "#e63946", "saddle": "#f4a261", "plane": "#f1fa3b", "condyloid": "#c77dff", "costovertebral": "#4cc9f0"}


def group_of(n):
    g = rig.bone_group(n)
    if n.startswith(("humerus", "scapula", "clavicula")): return "arm"
    if n.startswith("femur"): return "leg"
    return g


def view(P, which):
    P = np.asarray(P)
    if which == "front":
        return np.stack([P[:, 0], -P[:, 1]], 1), P[:, 2]                   # person-left on the right of the image (viewer facing the person); y is down so negate
    return np.stack([P[:, 2], -P[:, 1]], 1), -P[:, 0]                      # side: face to the left


def draw(ax, atlas, pos, which, F_joints=None, show_chain=True, title=""):
    polys, depth, cols = [], [], []
    light = np.array([-0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    for s in atlas.structures:
        n = s["name"]
        if s["bind"] == "soft" or n == "brain":
            continue
        V = pos[n]; Fc = np.asarray(s["faces"]).reshape(-1, 3)
        tri = V[Fc]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        sh = 0.55 + 0.45 * np.abs(nrm @ light)
        base = np.array(matplotlib.colors.to_rgb(GROUP_COL[group_of(n)]))
        P2, d = view(tri.reshape(-1, 3), which)
        polys.append(P2.reshape(-1, 3, 2)); depth.append(d.reshape(-1, 3).mean(1)); cols.append(base[None] * sh[:, None])
    polys = np.concatenate(polys); depth = np.concatenate(depth); cols = np.concatenate(cols)
    order = np.argsort(-depth) if which == "front" else np.argsort(-depth)
    ax.add_collection(PolyCollection(polys[order], facecolors=cols[order], edgecolors="none", linewidths=0))
    if show_chain:
        ch = [pos[n].mean(0) for n in atlas.meta["chain"]]; C, _ = view(np.array(ch), which)
        ax.plot(C[:, 0], C[:, 1], "-", color="#222", lw=1.2, alpha=0.9, zorder=5)
        ax.scatter(C[:, 0], C[:, 1], s=7, color="#222", zorder=6)
    for j in atlas.meta["joints"]:
        c = F_joints[j["name"]] if F_joints is not None else np.asarray(j["center"])
        q, _ = view(c[None], which)
        t = j["type"]
        ax.scatter(q[:, 0], q[:, 1], s=(10 if t == "costovertebral" else 55), color=JOINT_COL[t], edgecolors="k", linewidths=0.6, zorder=8)
    allp = np.concatenate([view(v, which)[0] for n, v in pos.items() if n in {s["name"] for s in atlas.structures if s["bind"] != "soft"}])
    ax.set_xlim(allp[:, 0].min() - 0.05, allp[:, 0].max() + 0.05); ax.set_ylim(allp[:, 1].min() - 0.05, allp[:, 1].max() + 0.05)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)


def tree_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 7)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(-0.6, 6.2)
    def box(x, y, t, c, w=1.5, h=0.5): ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc=c, ec="k", lw=0.8)); ax.text(x, y, t, ha="center", va="center", fontsize=8.5)
    def link(a, b, t="", col="#444", style="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-", color=col, lw=1.4, ls=style))
        if t: ax.text((a[0] + b[0]) / 2 + 0.05, (a[1] + b[1]) / 2, t, fontsize=7, color=col)
    sp = [("pelvis", GROUP_COL["pelvis"]), ("L1-L5", GROUP_COL["lumbar"]), ("T1-T12", GROUP_COL["thorax"]), ("C1-C7", GROUP_COL["head_neck"]), ("skull + brain", GROUP_COL["head_neck"])]
    ys = [0.7, 1.9, 3.1, 4.3, 5.5]
    for (t, c), y in zip(sp, ys): box(3.2, y, t, c)
    for a, b, l in zip(ys[:-1], ys[1:], ["lumbosacral", "thoracolumbar (group junction)", "cervicothoracic (group junction)", "atlanto-occipital (condyloid, limit 75 deg)"]):
        link((3.2, a + 0.25), (3.2, b - 0.25), l, JOINT_COL["condyloid"] if "atlanto" in l else "#444")
    box(6.0, 3.1, "ribs 1-12 (24)", GROUP_COL["thorax"], 1.3); link((3.95, 3.1), (5.35, 3.1)); ax.text(4.65, 3.22, "costovertebral hinge\n+ breathing", fontsize=7, color="#1790b8", ha="center", va="bottom")
    box(8.3, 3.1, "sternum", GROUP_COL["thorax"], 1.2); link((6.65, 3.1), (7.7, 3.1)); ax.text(7.15, 3.22, "carried by\nribs 2-7", fontsize=7, color="#444", ha="center", va="bottom")
    box(8.3, 2.1, "costal cartilage", GROUP_COL["thorax"], 1.6); link((8.3, 2.85), (8.3, 2.35), "flexible link: ends follow rib + sternum", "#0a7")
    box(8.3, 4.1, "clavicles", GROUP_COL["arm"], 1.2); link((8.3, 3.35), (8.3, 3.85), "sternoclavicular (saddle)", JOINT_COL["saddle"])
    box(8.3, 5.0, "scapulae", GROUP_COL["arm"], 1.2); link((8.3, 4.35), (8.3, 4.75), "acromioclavicular + glide on thorax", JOINT_COL["plane"])
    box(10.6, 5.0, "humeri", GROUP_COL["arm"], 1.2); link((8.9, 5.0), (10.0, 5.0)); ax.text(9.45, 5.1, "glenohumeral (ball)", fontsize=7, color=JOINT_COL["ball"], ha="center", va="bottom")
    box(0.9, 0.7, "femora", GROUP_COL["leg"], 1.2); link((2.45, 0.7), (1.5, 0.7)); ax.text(2.0, 0.8, "hip (ball)", fontsize=7, color=JOINT_COL["ball"], ha="center", va="bottom")
    ax.text(6.0, -0.35, "Rigid bones. Pinned joints: the child rotates about the joint centre and that centre is pinned to the parent's image of it, within limits.\n"
            "Soft tissue (organs, muscle, vessels): linear-blend skinning between the vertebral chain frames.\nInputs: 11 rigid segments (pelvis, lumbar, lower and upper thorax, head, shoulders, upper arms, thighs) estimated from the skin by weighted Kabsch.",
            ha="center", fontsize=8)
    ax.text(0.2, 6.0, "Kinematic structure of the rig (atlas v8)", fontsize=12, weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    atlas = Atlas(os.environ["ATLAS"]); rest = {s["name"]: s["pos"] for s in atlas.structures}
    fig, axs = plt.subplots(1, 2, figsize=(10, 9))
    draw(axs[0], atlas, rest, "front", title="Rest: front"); draw(axs[1], atlas, rest, "side", title="Rest: side (face left)")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", mfc=c, mec="k", label=k) for k, c in JOINT_COL.items()] + [Line2D([], [], color="#222", label="vertebral chain")]
    fig.legend(handles=leg + [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ()], loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("Rig at rest: bones by articulated group (head+neck, thorax+arms, lumbar, pelvis+legs), joints by type", fontsize=11)
    fig.savefig(f"{out}/rig_rest.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    if len(sys.argv) > 3:
        import onnxruntime as ort
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
        from mhr_clip import load_clip, prepare
        arr, fps, src = load_clip(sys.argv[2]); p = prepare(arr)[int(sys.argv[3])][None].astype(np.float32)
        sess = ort.InferenceSession(os.environ["ONNX"], providers=["CPUExecutionProvider"])
        skin = sess.run(None, {"mhr_params": p, "shape_params": np.zeros((1, 45), np.float32)})[1][0].astype(float)
        posed = atlas.pose(skin, spine="physio"); F = atlas.frames(skin, 0.0, 1.0, True, "physio")[3]
        Fj = {j["name"]: F[j["child"]][3] for j in atlas.meta["joints"] if j["child"] in F}
        fig, axs = plt.subplots(1, 2, figsize=(10, 9))
        draw(axs[0], atlas, posed, "front", Fj, title="Posed: front"); draw(axs[1], atlas, posed, "side", Fj, title="Posed: side")
        fig.suptitle(f"Rig posed by one frame of {os.path.basename(sys.argv[2])}: joint centres stay on their parent bones", fontsize=11)
        fig.savefig(f"{out}/rig_posed.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    tree_diagram(f"{out}/rig_tree.png"); print("done")
