"""Fit the body mesh to a depth frame: distance, articulated pose, then the surface itself.

The RGB model gives a posed body mesh whose outline is right in the picture but whose distance, lean and limb depths are a
guess. The depth camera measures the surface in front of it to a few millimetres. The fit takes the mesh to the measurement
in three stages, each explaining what the previous one could not:

  1 DISTANCE    every body part gets a scale about the camera (a point slides along its ray; the outline does not move): the
                weighted mean of measured depth / mesh depth over the part's visible vertices; a part with no evidence takes
                its parent's.
  2 POSE        articulated point-to-plane ICP. Each of the 19 parts gets a small rigid motion; a vertex moves with the blend
                of its parts' motions (the body model's skinning weights, summed per part). Solved together by Gauss-Newton,
                with a term that keeps each part attached to its parent where they meet and a damping term for parts the
                camera cannot see.
  3 SURFACE     what is left is shape (clothing, build, breathing). Each visible vertex's remaining error along its ray is
                spread over the mesh by a few rounds of smoothing (the measurement pulls where there is one, neighbours carry
                it into what the camera cannot see, fading with distance) and applied.

By default stages 1 and 2 are done for the body AS ONE (BodyFit.fit, articulated=False): one scale, one small rotation, one
translation. The pose stays the RGB model's. Per-part fitting, as described above, is available (articulated=True) but is not
robust and is not what the page uses; see BodyFit.fit. The skeleton goes through stages 1 and 2; stage 3 moves the skin only.

The depth is used as the camera delivers it. A vertex is carried into the depth camera's frame, projected through the depth
lens model to the pixel it falls on, and compared with the 3-D point that pixel measures (its ray times its depth, brought
back to the colour frame). Nothing is resampled, so a fit costs a few tens of milliseconds, and no measurement is invented
between pixels. Pixels on a depth edge (where a time-of-flight camera mixes two surfaces) are not used.

Frames: the mesh is in the COLOUR camera's frame, x right, y down, z forward, millimetres.
"""
import numpy as np
from scipy.sparse import csr_matrix

# body parts, as (name, parent, joint-name prefixes); a joint goes to the part with the longest prefix it starts with
PARTS = [
    ("pelvis", None, ("body_world", "root")),
    ("spine", "pelvis", ("c_spine0", "c_spine1")),
    ("chest", "spine", ("c_spine2", "c_spine3")),
    ("neck", "chest", ("c_neck",)),
    ("head", "neck", ("c_head", "c_jaw", "c_teeth", "c_tongue", "r_eye", "l_eye")),
    ("clavicle_l", "chest", ("l_clavicle",)), ("clavicle_r", "chest", ("r_clavicle",)),
    ("uparm_l", "clavicle_l", ("l_uparm",)), ("uparm_r", "clavicle_r", ("r_uparm",)),
    ("lowarm_l", "uparm_l", ("l_lowarm", "l_wrist_twist")), ("lowarm_r", "uparm_r", ("r_lowarm", "r_wrist_twist")),
    ("hand_l", "lowarm_l", ("l_wrist", "l_pinky", "l_ring", "l_middle", "l_index", "l_thumb")),
    ("hand_r", "lowarm_r", ("r_wrist", "r_pinky", "r_ring", "r_middle", "r_index", "r_thumb")),
    ("upleg_l", "pelvis", ("l_upleg",)), ("upleg_r", "pelvis", ("r_upleg",)),
    ("lowleg_l", "upleg_l", ("l_lowleg",)), ("lowleg_r", "upleg_r", ("r_lowleg",)),
    ("foot_l", "lowleg_l", ("l_foot", "l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball")),
    ("foot_r", "lowleg_r", ("r_foot", "r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball")),
]


def joint_parts(joint_names):
    """Part index of every joint."""
    out = np.zeros(len(joint_names), np.int64)
    for j, name in enumerate(joint_names):
        best = max(((len(p), g) for g, (_, _, pre) in enumerate(PARTS) for p in pre if name.startswith(p)), default=(0, 0))
        out[j] = best[1]
    return out


class DepthFrame:
    """One depth image with what is needed to read 3-D points off it.

    depth: (h, w) uint16 mm. rays: (h, w, 2), the direction (x/z, y/z) each pixel looks along. R, t: depth -> colour frame
    (mm). kd = (fx, fy, cx, cy) and dist8 = (k1, k2, p1, p2, k3, k4, k5, k6): the depth lens model, to find the pixel a point
    falls on."""

    def __init__(self, depth, rays, R, t, kd, dist8, flying_mm=30.0):
        self.depth = np.asarray(depth)
        self.rays, self.R, self.t = rays, np.asarray(R, np.float64).reshape(3, 3), np.asarray(t, np.float64)
        self.Kd = np.array([[kd[0], 0, kd[2]], [0, kd[1], kd[3]], [0, 0, 1.0]])
        self.dist = np.asarray(dist8, np.float64)
        self.flying_mm = flying_mm
        self.h, self.w = self.depth.shape

    def pixels(self, X):
        """Integer pixel (u, v) each colour-frame point falls on in the depth image, and whether it is inside it."""
        import cv2
        Xd = (X - self.t) @ self.R                               # R^T (x - t)
        front = Xd[:, 2] > 1.0
        uv = np.zeros((len(X), 2))
        if front.any():
            uv[front] = cv2.projectPoints(Xd[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.Kd, self.dist)[0].reshape(-1, 2)
        u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
        inside = front & (u >= 2) & (u < self.w - 2) & (v >= 2) & (v < self.h - 2)
        return u, v, inside, Xd[:, 2]

    def measure(self, u, v, inside, normals=False):
        """The 3-D point (colour frame) measured at each pixel, validity, and optionally the surface normal there (towards the
        camera). A pixel next to a jump in depth is not valid."""
        n = len(u)
        q = np.zeros((n, 3)); ok = np.zeros(n, bool); nrm = np.zeros((n, 3)) if normals else None
        idx = np.where(inside)[0]
        if len(idx) == 0:
            return q, ok, nrm
        ui, vi = u[idx], v[idx]
        D = self.depth
        d = D[vi, ui].astype(np.float64)
        nb = [D[vi, ui + 1].astype(np.float64), D[vi, ui - 1].astype(np.float64), D[vi + 1, ui].astype(np.float64), D[vi - 1, ui].astype(np.float64)]
        good = d > 0
        for x in nb:
            good &= (x > 0) & (np.abs(x - d) < self.flying_mm)
        for du, dv in ((2, 0), (-2, 0), (0, 2), (0, -2)):        # the mixed band at an edge is two or three pixels wide
            x = D[vi + dv, ui + du].astype(np.float64)
            good &= (x > 0) & (np.abs(x - d) < 1.5 * self.flying_mm)
        r = self.rays[vi, ui].astype(np.float64)
        pd = np.column_stack([r[:, 0] * d, r[:, 1] * d, d])
        q[idx] = pd @ self.R.T + self.t
        ok[idx] = good
        if normals:
            def pt(du, dv, dd):
                rr = self.rays[vi + dv, ui + du].astype(np.float64)
                return np.column_stack([rr[:, 0] * dd, rr[:, 1] * dd, dd])
            nn = np.cross(pt(1, 0, nb[0]) - pt(-1, 0, nb[1]), pt(0, 1, nb[2]) - pt(0, -1, nb[3]))
            ln = np.linalg.norm(nn, axis=1)
            nn = nn / np.maximum(ln, 1e-9)[:, None]
            nn[nn[:, 2] > 0] *= -1.0                             # towards the camera (depth frame)
            nrm[idx] = nn @ self.R.T
            ok[idx] &= ln > 1e-9
        return q, ok, nrm


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


class BodyFit:
    def __init__(self, faces, joint_names, w_joint, w_vert, w, n0=25.0, max_step=0.15, pose_points=2500):
        self.faces = np.asarray(faces, np.int64)
        self.jpart = joint_parts(list(joint_names))
        nv, G = int(np.max(w_vert)) + 1, len(PARTS)
        self.nv, self.G = nv, G
        W = csr_matrix((np.asarray(w, np.float64), (np.asarray(w_vert), self.jpart[np.asarray(w_joint)])), shape=(nv, G)).toarray()
        self.W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)   # (V, G) skinning weight of each part, rows sum to one
        names = [p[0] for p in PARTS]
        self.parent = np.array([-1 if p[1] is None else names.index(p[1]) for p in PARTS])
        self.n0, self.max_step, self.pose_points = n0, max_step, pose_points
        self.part_verts = [(np.where(self.W[:, g] > 0)[0], self.W[self.W[:, g] > 0, g][:, None]) for g in range(G)]
        # where a part meets its parent: the vertices both of them carry
        self.seam = [np.where((self.W[:, g] > 0.15) & (self.W[:, self.parent[g]] > 0.15))[0] if self.parent[g] >= 0 else np.array([], np.int64)
                     for g in range(G)]
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
        A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)); A.data[:] = 1.0
        self.A = csr_matrix(A.multiply(1.0 / np.maximum(np.asarray(A.sum(1)), 1.0)))    # neighbour average
        nf = len(f)
        self.F2V = csr_matrix((np.ones(nf * 3), (f.ravel(), np.repeat(np.arange(nf), 3))), shape=(nv, nf))   # face -> its vertices

    # ---- which vertices the depth camera sees, and what it measures there
    def observe(self, X, frame, normals=False, tol_mm=40.0):
        """-> dict: q (measured points), ok (visible, measured), n (depth normals or None), z (vertex depth, colour frame)."""
        f = self.faces
        fn = np.cross(X[f[:, 1]] - X[f[:, 0]], X[f[:, 2]] - X[f[:, 0]])
        vn = self.F2V @ fn                                        # area-weighted vertex normals (not normalised)
        facing = np.where(np.einsum("ij,ij->i", vn, X - frame.t) < 0.0)[0]   # towards the depth camera, which sits at t here
        nV = len(X)
        u = np.zeros(nV, np.int64); v = np.zeros(nV, np.int64); vis = np.zeros(nV, bool)
        uf, vf, inside, zd = frame.pixels(X[facing])
        ii = facing[inside]; ui, vi, zi = uf[inside], vf[inside], zd[inside]
        u[ii], v[ii] = ui, vi
        # self-occlusion: a vertex is hidden if another one, clearly nearer, falls in the 3 x 3 pixels around it
        if not hasattr(self, "_zbuf") or self._zbuf.shape != (frame.h, frame.w):
            self._zbuf = np.full((frame.h, frame.w), np.inf)
        zbuf = self._zbuf
        order = np.argsort(-zi)                                   # far first, so the nearest is written last
        zbuf[vi[order], ui[order]] = zi[order]
        near = zbuf[vi, ui].copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    np.minimum(near, zbuf[vi + dy, ui + dx], out=near)
        zbuf[vi, ui] = np.inf                                     # leave the buffer empty for the next call
        vis[ii] = zi <= near + tol_mm
        q, ok, n = frame.measure(u, v, vis, normals)
        return {"q": q, "ok": ok, "n": n, "vis": vis}

    # ---- stage 1
    def part_scales(self, X, frame, use=None):
        o = self.observe(X, frame)
        ok = o["ok"] if use is None else o["ok"] & use
        if ok.sum() < 300:
            return None, {"reason": "the body is not in the depth image", "samples": int(ok.sum())}
        r = np.zeros(len(X)); r[ok] = o["q"][ok, 2] / X[ok, 2]
        g = float(np.median(r[ok]))
        if not 0.5 < g < 1.8:
            return None, {"reason": f"implausible scale {g:.2f}", "samples": int(ok.sum())}
        ok = ok & (np.abs(r - g) < 0.25 * g)
        m = ok.astype(np.float64)
        n_g = self.W.T @ m
        mean_g = (self.W.T @ (m * r)) / np.maximum(n_g, 1e-9)
        s = np.empty(self.G)
        for gi in range(self.G):                                 # PARTS lists parents before children
            prior = g if self.parent[gi] < 0 else s[self.parent[gi]]
            c = n_g[gi] / (n_g[gi] + self.n0)
            s[gi] = np.clip(c * mean_g[gi] + (1 - c) * prior, prior * (1 - self.max_step), prior * (1 + self.max_step))
        return s, {"global_scale": g, "samples": int(ok.sum()), "evidence": n_g}

    # ---- stage 2
    def pose_step(self, X, frame, use=None, max_mm=60.0, damp=2.0, seam_w=3.0, keep=None):
        """One Gauss-Newton step of articulated point-to-plane ICP at the mesh X.
        -> (increment of each part as (G, 4, 4), number of vertices used) or (None, n)."""
        G, Wn = self.G, self.W
        o = self.observe(X, frame, normals=True)
        if keep is not None:
            keep["o"] = o                                        # the caller may reuse what was measured
        ok = o["ok"] & (np.linalg.norm(o["q"] - X, axis=1) < max_mm)
        if use is not None:
            ok &= use
        c = (Wn.T @ X) / np.maximum(Wn.sum(0), 1e-9)[:, None]     # part centres (weighted centroids)
        idx = np.where(ok)[0]
        if len(idx) < 300:
            return None, int(len(idx))
        if len(idx) > self.pose_points:
            idx = idx[np.linspace(0, len(idx) - 1, self.pose_points).astype(np.int64)]
        Xo, no = X[idx], o["n"][idx]
        r = np.einsum("ij,ij->i", no, Xo - o["q"][idx])
        wr = 1.0 / np.maximum(1.0, np.abs(r) / 10.0)             # Huber weights, 10 mm
        # Unknowns: the parts the camera has evidence for, and their ancestors. Any other part moves with its nearest such
        # ancestor (legs under a desk follow the pelvis).
        Wi = Wn[idx]
        active = Wi.sum(0) > 2.0
        for g in range(G - 1, -1, -1):
            if active[g] and self.parent[g] >= 0:
                active[self.parent[g]] = True
        act = np.where(active)[0]
        col = -np.ones(G, np.int64); col[act] = np.arange(len(act))
        n6 = 6 * len(act)
        J = np.zeros((len(idx), n6))
        for g in act:
            k, wg = 6 * col[g], Wi[:, g:g + 1]
            a = Xo - c[g]
            J[:, k] = wg[:, 0] * (a[:, 1] * no[:, 2] - a[:, 2] * no[:, 1])
            J[:, k + 1] = wg[:, 0] * (a[:, 2] * no[:, 0] - a[:, 0] * no[:, 2])
            J[:, k + 2] = wg[:, 0] * (a[:, 0] * no[:, 1] - a[:, 1] * no[:, 0])
            J[:, k + 3:k + 6] = wg * no
        H = (J * wr[:, None]).T @ J
        b = -J.T @ (wr * r)
        scale = len(idx) / 1000.0
        # Damping, per part: strong where the camera sees little of the part (its motion is then set by the seam to its
        # parent), light where it sees a lot, so a well-observed part reaches the measurement in one or two steps instead
        # of creeping towards it. Rotations in radians are small numbers, translations in mm large ones.
        ev = Wi.sum(0)[act]
        dg = damp * scale * np.maximum(0.03, 40.0 / (ev + 40.0))
        H[np.diag_indices(n6)] += (dg[:, None] * np.array([2.5e5] * 3 + [1.0] * 3)[None]).ravel()
        # seams: where a part meets its parent the two motions must give the same displacement
        rows = []
        for g in act:
            p, sv = self.parent[g], self.seam[g]
            if p < 0 or len(sv) == 0:
                continue
            j = X[sv].mean(0)
            kg, kp = 6 * col[g], 6 * col[p]
            for axis in range(3):
                row = np.zeros(n6); e = np.zeros(3); e[axis] = 1.0
                row[kg:kg + 3] = np.cross(j - c[g], e); row[kg + 3:kg + 6] = e
                row[kp:kp + 3] -= np.cross(j - c[p], e); row[kp + 3:kp + 6] -= e
                rows.append(row)
        if rows:
            S = np.array(rows)
            H += (seam_w * scale * 50.0) * (S.T @ S)
        from scipy.linalg import cho_factor, cho_solve
        delta = cho_solve(cho_factor(H, check_finite=False), b, check_finite=False)
        D = np.tile(np.eye(4), (G, 1, 1))                        # increment of each part: x -> R (x - c) + c + t
        for g in range(G):                                       # parents come first in PARTS
            if active[g]:
                k = 6 * col[g]
                R, t = _rodrigues(delta[k:k + 3]), delta[k + 3:k + 6]
                D[g, :3, :3] = R; D[g, :3, 3] = c[g] - R @ c[g] + t
            elif self.parent[g] >= 0:
                D[g] = D[self.parent[g]]                         # no evidence: move with the parent
        return D, int(len(idx))

    def global_step(self, X, frame, use=None, w_acc=None, max_mm=60.0, damp=0.3, rot_prior=10.0, keep=None):
        """One Gauss-Newton step of rigid point-to-plane ICP for the WHOLE body: one rotation and one translation.
        w_acc: the rotation already applied (rotation vector); the step is pulled back towards no rotation at all.
        -> (the increment as a 4 x 4 matrix, number of vertices used) or (None, n)."""
        o = self.observe(X, frame, normals=True)
        if keep is not None:
            keep["o"] = o
        ok = o["ok"] & (np.linalg.norm(o["q"] - X, axis=1) < max_mm)
        if use is not None:
            ok &= use
        idx = np.where(ok)[0]
        if len(idx) < 300:
            return None, int(len(idx))
        if len(idx) > self.pose_points:
            idx = idx[np.linspace(0, len(idx) - 1, self.pose_points).astype(np.int64)]
        Xo, no = X[idx], o["n"][idx]
        r = np.einsum("ij,ij->i", no, Xo - o["q"][idx])
        wr = 1.0 / np.maximum(1.0, np.abs(r) / 10.0)             # Huber weights, 10 mm
        c = Xo.mean(0)
        a = Xo - c
        J = np.column_stack([a[:, 1] * no[:, 2] - a[:, 2] * no[:, 1], a[:, 2] * no[:, 0] - a[:, 0] * no[:, 2],
                             a[:, 0] * no[:, 1] - a[:, 1] * no[:, 0], no])
        H = (J * wr[:, None]).T @ J
        b = -J.T @ (wr * r)
        scale = len(idx) / 1000.0
        H[np.diag_indices(6)] += damp * scale * np.array([2.5e5] * 3 + [1.0] * 3)
        if w_acc is not None:
            # The body's orientation is the picture's to say: a depth image holds a turn poorly (a head is nearly a ball,
            # a trunk nearly a cylinder). The total rotation is therefore pulled towards none, and the depth turns the
            # body only as far as it insists.
            lam = rot_prior * scale * 2.5e5
            H[:3, :3] += lam * np.eye(3)
            b[:3] -= lam * np.asarray(w_acc)
        delta = np.linalg.solve(H, b)
        R = _rodrigues(delta[:3])
        D = np.eye(4); D[:3, :3] = R; D[:3, 3] = c - R @ c + delta[3:]
        return D, int(len(idx))

    def blend(self, V, M):
        """Vertices carried by their parts' transforms M (G, 4, 4), blended by the skinning weights."""
        X = np.zeros_like(V)
        for g, (idx, wg) in enumerate(self.part_verts):          # only the vertices each part carries
            if len(idx):
                X[idx] += wg * (V[idx] @ M[g, :3, :3].T + M[g, :3, 3])
        return X

    # ---- stage 3
    def surface_step(self, X, frame, use=None, max_mm=40.0, near_mm=80.0, rounds=12, pull=1.0, o=None, track=None, alpha=0.5):
        """Spread each visible vertex's remaining error along its ray over the mesh.
        -> X', number of vertices measured, their remaining error along the ray (mm)."""
        if o is None:
            o = self.observe(X, frame)
        dz = o["q"][:, 2] - X[:, 2]
        ok = o["ok"] & (dz < max_mm) & (dz > -near_mm)            # a measurement in front (cloth, a hand) is followed further
        if use is not None:                                      # than one behind (the background past the outline)
            ok &= use
        e = np.zeros(len(X)); e[ok] = (o["q"][ok, 2] - X[ok, 2]) / X[ok, 2]     # relative error along the ray
        m = ok.astype(np.float64) * pull
        u = e.copy()
        for _ in range(rounds):                                  # u ~ argmin pull*m*(u - e)^2 + (u - neighbour average)^2
            u = (self.A @ u + m * e) / (1.0 + m)
        if track is not None:
            # Over time: the offsets are averaged with the previous frame's (the depth camera's noise is new every frame, the
            # shape it measures is not). Bulk motion is the pose stage's and is not delayed by this.
            if track.get("u") is not None:
                u = (1.0 - alpha) * track["u"] + alpha * u
            track["u"] = u
        return X * (1.0 + u)[:, None], int(ok.sum()), (e[ok] - u[ok]) * X[ok, 2]      # what is left at the measured vertices, mm

    # ---- all three
    def fit(self, V, frame, use=None, init=None, pose_iters=2, surface=True, dead_mm=0.8, track=None, articulated=False):
        """V: (nv, 3) mesh in the colour camera's frame, mm. use: optional mask of the vertices allowed to see the depth (for
        testing). init: the transforms of a previous fit of (nearly) this mesh, to start from. track: a dict kept by the
        caller from frame to frame (surface offsets averaged over time; it also receives "X_pose", the mesh before stage 3).

        articulated=False (the default): the body is placed AS ONE. Its pose is the RGB model's, untouched; the depth
        gives one scale about the camera, one small rotation and one translation for all of it, and then the surface
        stage. Every part has the same transform, so nothing can come apart or turn on its own.
        articulated=True: each body part gets its own distance and its own rigid motion (stages 1 and 2 as described at
        the top). It fits a little closer and is NOT robust: a part the depth holds poorly (a head, against turning) is
        rotated by noise, and a correction worked out for one pose is wrong for the next. Kept for experiments.

        -> (fitted vertices, transforms M (G, 4, 4): the similarity of each part, or None, info)."""
        info = {"ok": False}
        if init is None:
            s, a = self.part_scales(V, frame, use)
            if s is None:
                info.update(a)
                return V, None, info
            info["global_scale"] = round(a["global_scale"], 4)
            M = np.tile(np.eye(4), (self.G, 1, 1))
            M[:, :3, :3] *= (s if articulated else np.full(self.G, a["global_scale"]))[:, None, None]
        else:
            M = np.array(init, np.float64)
        X = self.blend(V, M) if articulated else V @ M[0, :3, :3].T + M[0, :3, 3]
        n_pose, seen, moved = 0, {}, 0.0
        for _ in range(pose_iters):
            if articulated:
                D, n = self.pose_step(X, frame, use, keep=seen)
            else:
                from scipy.spatial.transform import Rotation
                Rm = M[0, :3, :3] / np.cbrt(np.linalg.det(M[0, :3, :3]))
                D, n = self.global_step(X, frame, use, w_acc=Rotation.from_matrix(Rm).as_rotvec(), keep=seen)
            if D is None:
                if init is not None and n_pose == 0:
                    info["reason"] = "the body is not in the depth image"
                    return V, None, info
                break
            Mn = D @ M
            Xn = self.blend(V, Mn) if articulated else V @ Mn[0, :3, :3].T + Mn[0, :3, 3]
            ov = seen["o"]["ok"]                                  # how far this step would carry what the camera sees, mm
            moved = float(np.percentile(np.abs(Xn[ov] - X[ov]).max(1), 95)) if ov.any() else 0.0
            n_pose += 1
            if init is not None and moved < dead_mm:
                # Dead band. Point-to-plane ICP never quite stops: along a surface the data does not hold the body, and it
                # creeps by a fraction of a millimetre a frame, which shows as shimmer and as skin sliding over the
                # anatomy. A step this small is not taken; what it would have corrected along the ray, stage 3 corrects.
                moved = 0.0
                break
            M, X = Mn, Xn
        if track is not None:
            track["X_pose"] = X                                   # the placed mesh, before the skin is drawn onto the surface
        if surface:
            # When tracking at the camera's rate one pose step moves the mesh by less than a depth pixel (about 1 mm at half
            # a metre), so the measurements it was made from serve the surface stage too and the depth image is read once.
            reuse = seen.get("o") if (init is not None and n_pose == 1 and moved < 1.0) else None      # (moved is 0 in the dead band)
            info["reused"] = reuse is not None
            info["moved_mm"] = round(moved, 2) if n_pose else None
            X, n, left = self.surface_step(X, frame, use, o=reuse, track=track)
            info["surface_vertices"] = n
            if n:
                info["median_mm"] = round(float(np.median(np.abs(left))), 2)
                info["within_2mm"] = round(float((np.abs(left) < 2.0).mean()), 3)
        info.update(ok=True, pose_steps=n_pose, part_scale=[round(float(np.cbrt(np.linalg.det(M[g, :3, :3]))), 3) for g in (0, 2, 4)])
        return X, M, info

    def residual(self, X, frame, mask=None, max_mm=80.0):
        """Signed error along the ray (mm) at the visible, measured vertices (optionally only those in mask)."""
        o = self.observe(X, frame)
        e = o["q"][:, 2] - X[:, 2]
        ok = o["ok"] & (np.abs(e) < max_mm)
        if mask is not None:
            ok &= mask
        return e[ok]

    def move_joints(self, pos, rot, M):
        """Joint positions (colour frame, mm) and rotations carried by their part's transform. -> (positions, rotations)."""
        Mj = M[self.jpart]
        P = np.einsum("jab,jb->ja", Mj[:, :3, :3], np.asarray(pos, np.float64)) + Mj[:, :3, 3]
        if rot is None:
            return P, None
        sc = np.cbrt(np.linalg.det(Mj[:, :3, :3]))
        return P, np.einsum("jab,jbc->jac", Mj[:, :3, :3] / sc[:, None, None], np.asarray(rot, np.float64))
