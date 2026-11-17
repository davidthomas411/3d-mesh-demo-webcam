"""Deform the atlas subject's REAL CT with the anatomy rig's motion, then render radiographs and reformats.

What this is: a dense displacement field built from how every structure in the rig moves (bones: exact rigid
motion in their own CT frame; soft tissue: the skinned motion), interpolated by inverse-distance weighting and
applied by backward warping. It gives a deformed CT with a known voxel-to-voxel correspondence to the original.
What it is NOT: a physical simulation. Tissue far from any modelled structure is interpolated; sliding,
breathing and gas are not modelled; arms and couch are removed.

Frames: everything in the CT's own frame, millimetres, MHR axes (x = patient left, y = down, z = back).
"""
import os
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from anatomy import atlas_build as ab
from anatomy import rig
from anatomy.runtime import Atlas

SKIP_LANDMARKS = ("humerus_", "femur_")       # arms/legs are cropped; their CT pose differs from the atlas T-pose
RIGID_WITH = {"brain": "skull"}              # soft structures that move as one rigid body with a bone (brain with the skull)
RIGID_SOFT = tuple(RIGID_WITH)


class CTWarper:
    def __init__(self, subject_dir, corr_npz, atlas_prefix, mhr_rest, margin_mm=35.0):
        self.atlas = Atlas(atlas_prefix)
        c = np.load(corr_npz)
        self.names = [str(n) for n in c["names"]]
        self.v_ct = {n: c[f"v_{n}"] for n in self.names}
        Tb = {n: c[f"T_{n}"] for n in self.names}
        self.T = {n: (Tb[n][0], Tb[n][1:10].reshape(3, 3), Tb[n][10:13]) for n in self.names}
        w = c["Wsim"]
        self.W = (float(w[0]), w[1:10].reshape(3, 3), w[10:13])
        self.by = {s["name"]: s for s in self.atlas.structures}
        self.mhr_rest = np.asarray(mhr_rest, float)
        self.skin_ct = c["skin_ct"][::6]; self.skin_nn = c["skin_nn"][::6]
        # ---- CT volume in MHR axes
        img = nib.load(f"{subject_dir}/ct.nii.gz")
        self.hu = np.asarray(img.dataobj).astype(np.int16)
        aff = img.affine
        self.A = ab.CT_TO_MHR @ aff[:3, :3]
        self.b = ab.CT_TO_MHR @ aff[:3, 3]
        self.Ainv = np.linalg.inv(self.A)
        _, masks = ab.load_subject(subject_dir)
        self.label = np.zeros(self.hu.shape, np.uint8)
        self.label_names = {}
        for k, (n, (m, lo)) in enumerate(sorted(masks.items()), start=1):
            sl = tuple(slice(l, l + s) for l, s in zip(lo, m.shape))
            self.label[sl][m] = k
            self.label_names[k] = n
        bone_ids = [k for k, n in self.label_names.items()
                    if ab.is_bone(n) and not n.startswith(SKIP_LANDMARKS)]
        self.bone = np.isin(self.label, bone_ids)
        self.bone = ndi.binary_dilation(self.bone, iterations=1)
        rigid_ids = bone_ids + [k for k, n in self.label_names.items() if n in RIGID_SOFT]   # brain rides rigidly with the skull
        bid = np.where(np.isin(self.label, rigid_ids), self.label, 0).astype(np.uint8)
        self.bone_id = ndi.grey_dilation(bid, size=3)                       # bone identity per voxel (for exact rigid motion)
        self.bone_near = ndi.binary_dilation(self.bone, iterations=3)       # candidates for the rigid override
        # ---- remove couch and arms
        self.keep = self._body_mask() & self._lateral_mask(margin_mm)
        # the couch/arm mask must never cut the skeleton or the brain it rides with (it removed 2945 skull voxels, 1.4%, at the occiput:
        # a visible hole in the skull whatever the pose)
        self.keep |= ndi.binary_dilation(np.isin(self.label, rigid_ids), iterations=2)
        self.hu_c = np.where(self.keep, self.hu, -1000).astype(np.int16)
        self.bone &= self.keep
        # CT skin landmarks that lie in the kept region (arms and couch excluded)
        ijk = np.rint(self.mm_to_vox(self.skin_ct)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.hu.shape)), 1)
        ok[ok] = self.keep[tuple(ijk[ok].T)]
        self.skin_ct, self.skin_nn = self.skin_ct[ok], self.skin_nn[ok]
        self.label_id = {n: k for k, n in self.label_names.items()}

    # ---------------------------------------------------------------- geometry helpers
    def vox_to_mm(self, ijk):
        return ijk @ self.A.T + self.b

    def mm_to_vox(self, p):
        return (p - self.b) @ self.Ainv.T

    def _body_mask(self):
        d = self.hu[::2, ::2, ::2] > -300
        d = ndi.binary_opening(d, structure=ndi.generate_binary_structure(3, 1), iterations=4)
        lab, n = ndi.label(d)
        d = lab == (1 + int(np.argmax(ndi.sum(d, lab, range(1, n + 1)))))
        for k in range(d.shape[2]):
            d[:, :, k] = ndi.binary_fill_holes(d[:, :, k])
        up = np.repeat(np.repeat(np.repeat(d, 2, 0), 2, 1), 2, 2)[: self.hu.shape[0], : self.hu.shape[1], : self.hu.shape[2]]
        return ndi.binary_dilation(up, iterations=3)

    def _lateral_mask(self, margin):
        """per axial slice, keep |x - spine| < (lateral extent of the axial skeleton) + margin: drops the arms."""
        pts = np.vstack([self.v_ct[n] for n in self.names if ab.is_bone(n) and not n.startswith(SKIP_LANDMARKS)])
        spine = np.vstack([self.v_ct[n] for n in self.names if n.startswith("vertebrae_")])
        xc = spine[:, 0].mean()
        ys = np.arange(pts[:, 1].min() - 100, pts[:, 1].max() + 100, 10.0)
        lim = np.array([np.abs(pts[(pts[:, 1] > y - 20) & (pts[:, 1] < y + 20), 0] - xc).max()
                        if ((pts[:, 1] > y - 20) & (pts[:, 1] < y + 20)).any() else np.nan for y in ys])
        ok = ~np.isnan(lim)
        lim = np.interp(ys, ys[ok], lim[ok]) if ok.any() else np.full_like(ys, 200.0)
        lim = ndi.uniform_filter1d(lim, 5, mode="nearest") + margin
        # the CT axes are signed permutations of the body axes, so x and y each depend on ONE voxel axis
        ax_x, ax_y = int(np.argmax(np.abs(self.A[0]))), int(np.argmax(np.abs(self.A[1])))
        x1 = self.A[0, ax_x] * np.arange(self.hu.shape[ax_x]) + self.b[0]
        y1 = self.A[1, ax_y] * np.arange(self.hu.shape[ax_y]) + self.b[1]
        okx = np.abs(x1 - xc)[:, None] < np.interp(y1, ys, lim)[None, :] if ax_x != ax_y else None
        shape = [1, 1, 1]
        mask = np.ones(self.hu.shape, bool)
        # broadcast the 2-D (x, y) table to the 3-D grid
        tab = okx if ax_x < ax_y else okx.T
        shp = [1, 1, 1]
        shp[min(ax_x, ax_y)] = tab.shape[0]; shp[max(ax_x, ax_y)] = tab.shape[1]
        return mask & tab.reshape(shp)

    # ---------------------------------------------------------------- the motion field
    @staticmethod
    def local_rotations(rest, posed, k=24, chunk=50000):
        """Local rotation of the deformation at every point of a corresponded cloud: Kabsch fit of each point's k nearest neighbours, rest -> posed."""
        tree = cKDTree(rest); R = np.empty((len(rest), 3, 3))
        for a in range(0, len(rest), chunk):
            sl = slice(a, a + chunk); idx = tree.query(rest[sl], k=k, workers=-1)[1]
            Pr, Pp = rest[idx], posed[idx]
            Pr = Pr - Pr.mean(1, keepdims=True); Pp = Pp - Pp.mean(1, keepdims=True)
            U, S, Vt = np.linalg.svd(np.einsum("nki,nkj->nij", Pr, Pp))
            Rm = np.einsum("nji,nkj->nik", Vt, U)                               # V U^T: maps rest-frame vectors to posed-frame vectors
            neg = np.linalg.det(Rm) < 0
            Vt2 = Vt.copy(); Vt2[neg, 2, :] *= -1
            Rm = np.einsum("nji,nkj->nik", Vt2, U)
            R[sl] = Rm
        return R

    def bone_ct_positions(self, name, posed_vertices):
        """Posed positions (CT frame, mm) of a bone's CT vertices. The atlas bone moves rigidly, p = Q r + tau; the CT bone (placed in the avatar frame by the one
        global registration W) takes the SAME rigid motion, y = Q W(x) + tau, exactly as skin and soft-tissue landmarks do (see local_rotations), so bone, skin and
        organs of the CT subject keep their own relative geometry whatever the pivot. (Two earlier versions moved the bone about its own centroid or through its own
        CT->atlas similarity; both added (I - Q) times the offset between the CT bone and its atlas bone, 50-90 mm at a 100 degree head turn.) At rest this returns
        the CT vertices unchanged."""
        s_w, R_w, t_w = self.W
        r = self.by[name]["pos"]; p = posed_vertices
        _, Q, _ = ab.umeyama(r, p, with_scale=False)                       # rigid fit of the posed atlas bone
        tau = p.mean(0) - Q @ r.mean(0)
        q = (self.v_ct[name] @ R_w.T) * s_w + t_w                           # the CT bone in the avatar frame under W
        ya = q @ Q.T + tau
        return ((ya - t_w) @ R_w) / s_w

    def skin_targets(self, skin_posed, bones, attach_mm=25.0, rigid_transfer=True):
        """Posed CT-frame positions of every CT skin landmark. Each CT skin point is carried by the LOCAL rotation of the posed MHR skin at its matched vertex
        (rigid_transfer; displacement-only transfer errs by (R - I) times the CT-to-MHR registration offset wherever the tissue rotates), then blended toward the
        rigid motion of the nearest bone where the skin lies on bone (attach_mm; 0 disables)."""
        s_w, R_w, t_w = self.W
        if rigid_transfer and os.environ.get("CTW_RT_SKIN", "1") == "1":
            Rs = self.local_rotations(self.mhr_rest, skin_posed)[self.skin_nn]
            q = (self.skin_ct @ R_w.T) * s_w + t_w
            ya = skin_posed[self.skin_nn] + np.einsum("nij,nj->ni", Rs, q - self.mhr_rest[self.skin_nn])
            Ys = ((ya - t_w) @ R_w) / s_w
        else:
            Ys = self.skin_ct + (skin_posed[self.skin_nn] - self.mhr_rest[self.skin_nn]) @ R_w / s_w
        if bones and attach_mm:
            bx = np.vstack([b[2] for b in bones.values()]); bk = np.concatenate([[k_] * len(b[2]) for k_, b in bones.items()])
            dist, ii = cKDTree(bx).query(self.skin_ct); owner = bk[ii]
            Yb = np.empty_like(Ys)
            for k_ in np.unique(owner):
                m = owner == k_; Rb, tb = bones[k_][:2]; Yb[m] = self.skin_ct[m] @ Rb.T + tb
            a = np.exp(-0.5 * (dist / attach_mm) ** 2)[:, None]
            Ys = a * Yb + (1 - a) * Ys
        return Ys

    def landmarks(self, posed, skin_posed, structures=True, offset=0, skin_half=False, soft=True, attach_mm=25.0, rigid_transfer=True):
        """-> X (source, CT frame mm), Y (posed, CT frame mm), and per-bone data for the exact rigid override."""
        s_w, R_w, t_w = self.W
        X, Y, bones = [], [], {}
        Rloc = None
        # Soft-tissue landmarks are carried by displacement only by default: carrying them with the local rotation (CTW_RT_SOFT=1) threw head tissue to the abdomen
        # and doubled the held-out error (the offset between the CT structure and its atlas counterpart is rotated, and the local rotation changes abruptly at joints).
        if structures and rigid_transfer and os.environ.get("CTW_RT_SOFT", "0") == "1":
            names_ = [n for n in self.names if n in posed and n in self.by]
            allr = np.vstack([self.by[n]["pos"] for n in names_]); allp = np.vstack([posed[n] for n in names_])
            Rall = self.local_rotations(allr, allp); Rloc, o = {}, 0
            for n in names_:
                Rloc[n] = Rall[o:o + len(self.by[n]["pos"])]; o += len(self.by[n]["pos"])
        for n in (self.names if structures else []):
            if n.startswith(SKIP_LANDMARKS):
                continue
            if not soft and not (ab.is_bone(n) or n in RIGID_SOFT):
                continue                                    # bones + skin only (no organ/vessel/muscle landmarks)
            xs = self.v_ct[n][offset::2]
            p = posed[n][offset::2]
            if n in RIGID_SOFT:
                continue                                    # given the carrier bone's own rigid motion below (not an independent fit)
            if ab.is_bone(n):
                ys = self.bone_ct_positions(n, posed[n])[offset::2]
                _, Rb, tb = ab.umeyama(xs, ys, with_scale=False)   # rigid motion of the bone in the CT frame: y = Rb x + tb
                bones[self.label_id[n]] = (Rb, tb, xs, ys)
            else:
                rest = self.by[n]["pos"][offset::2]
                if Rloc is None:
                    ys = xs + ((p - rest) @ R_w) / s_w      # displacement-only transfer (wrong where tissue rotates: error (R - I) * registration offset)
                else:
                    q = (xs @ R_w.T) * s_w + t_w            # this CT point in the avatar frame under the global registration
                    ya = p + np.einsum("nij,nj->ni", Rloc[n][offset::2], q - rest)   # carried by the LOCAL rotation of the atlas deformation there
                    ys = ((ya - t_w) @ R_w) / s_w
            X.append(xs); Y.append(ys)
        for n, carrier in RIGID_WITH.items():                # brain: exactly the skull's rigid motion in the CT frame (separate fits differ by ~3 deg / 15 mm)
            if n in self.names and self.label_id[carrier] in bones:
                Rb, tb = bones[self.label_id[carrier]][:2]
                xs = self.v_ct[n][::2]
                bones[self.label_id[n]] = (Rb, tb, xs, xs @ Rb.T + tb)
                X.append(xs); Y.append(xs @ Rb.T + tb)
        # the outer body follows the posed MHR skin
        du = (skin_posed[self.skin_nn] - self.mhr_rest[self.skin_nn]) @ R_w / s_w
        Ys = self.skin_targets(skin_posed, bones, attach_mm, rigid_transfer)
        sk = slice(0, None, 2) if skin_half else slice(None)      # skin_half: build the field from half the skin points (held-out test)
        X.append(self.skin_ct[sk]); Y.append(Ys[sk])
        return np.vstack(X), np.vstack(Y), bones

    @staticmethod
    def _mls_u(G, X, Y, idx, w, chunk=60000):
        """Moving-least-squares RIGID backward map: at each grid node the weighted Kabsch fit of the neighbouring posed landmarks Y onto their
        sources X gives a local rotation and translation, src(g) = R (g - ybar) + xbar; returns u = g - src. A weighted mean of displacement vectors
        (the old scheme) is only right where tissue translates; where it rotates (head turn, shoulder, bend) it tears."""
        u = np.empty_like(G)
        for a in range(0, len(G), chunk):
            sl = slice(a, a + chunk); ii = idx[sl]; ww = w[sl] / w[sl].sum(1, keepdims=True)
            Yi, Xi = Y[ii], X[ii]
            yb = (ww[..., None] * Yi).sum(1); xb = (ww[..., None] * Xi).sum(1)
            Hm = np.einsum("nk,nki,nkj->nij", ww, Yi - yb[:, None], Xi - xb[:, None])
            U, S, Vt = np.linalg.svd(Hm)
            R = np.einsum("nji,nkj->nik", Vt, U)                                   # V U^T  (rows: R maps Y-frame vectors to X-frame)
            dets = np.linalg.det(R)
            Vt2 = Vt.copy(); Vt2[:, 2, :] *= np.where(dets < 0, -1.0, 1.0)[:, None]
            R = np.einsum("nji,nkj->nik", Vt2, U)
            weak = S[:, 1] < 1e-3 * np.maximum(S[:, 0], 1e-12)                      # collinear support: rotation undefined, fall back to translation
            R[weak] = np.eye(3)
            g = G[sl]; src = np.einsum("nij,nj->ni", R, g - yb) + xb
            u[sl] = g - src
        return u

    def field(self, posed, skin_posed, grid_mm=6.0, k=32, sigma_mm=10.0, structures=True, method="shepard", skin_half=False, soft=True, attach_mm=25.0, rigid_transfer=True):
        """Coarse backward displacement field u_hat(g) for output (posed) positions g, plus the exact bone motions."""
        X, Y, bones = self.landmarks(posed, skin_posed, structures, skin_half=skin_half, soft=soft, attach_mm=attach_mm, rigid_transfer=rigid_transfer)
        U = Y - X
        tree = cKDTree(Y)
        lo = self.vox_to_mm(np.array([0, 0, 0.0])); hi = self.vox_to_mm(np.array(self.hu.shape, float) - 1)
        mn, mx = np.minimum(lo, hi), np.maximum(lo, hi)
        axes = [np.arange(mn[a], mx[a] + grid_mm, grid_mm) for a in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        if method == "mls":
            k, sigma_mm = max(k, 48), max(sigma_mm, 15.0)
        d, i = tree.query(G, k=k, workers=-1)
        w = np.exp(-(d ** 2 - d[:, :1] ** 2) / (2 * sigma_mm ** 2))        # Gaussian, relative to the nearest landmark
        if method == "mls":
            uh = self._mls_u(G, X, Y, i, w)
        else:
            uh = (w[..., None] * U[i]).sum(1) / w.sum(1, keepdims=True)    # weighted mean displacement: translation only, wrong where tissue rotates
        shape = tuple(len(a) for a in axes)
        # bone landmark tree for the rigid override
        if bones:
            if getattr(self, "_bone_pts", None) is None:      # bone voxel centres (stride 2), per bone label, in the CT frame (mm)
                sub = np.zeros(self.bone_id.shape, bool); sub[::2, ::2, ::2] = True
                ii = np.argwhere(sub & (self.bone_id > 0))
                lab_ = self.bone_id[tuple(ii.T)]
                self._bone_pts = {int(k_): self.vox_to_mm(ii[lab_ == k_].astype(float)) for k_ in np.unique(lab_)}
            pts = [(k_, self._bone_pts[k_]) for k_ in bones if k_ in self._bone_pts]
            bl = np.vstack([(bones[k_][0] @ P.T).T + bones[k_][1] for k_, P in pts])
            own = np.concatenate([np.full(len(P), k_) for k_, P in pts])
            bt = cKDTree(bl)
        else:                                       # skin-only mode: no skeleton, nothing rigid
            own, bt = None, None
        return {"origin": mn, "grid": grid_mm, "u": uh.reshape(shape + (3,)), "shape": shape,
                "bones": bones, "bone_tree": bt, "bone_owner": own}

    def source_positions(self, fld, g, override_mm=8.0):
        """CT positions to sample for output positions g (n,3 mm): smooth field everywhere, and inside a posed bone
        the bone's exact rigid inverse (so bones are never deformed, tissue around them is)."""
        c = ((g - fld["origin"]) / fld["grid"]).T
        u = np.stack([ndi.map_coordinates(fld["u"][..., a], c, order=1, mode="nearest") for a in range(3)], 1)
        src = g - u
        if fld["bone_tree"] is None:
            return src
        near = self.sample(self.bone_near.astype(np.uint8), src, order=0, cval=0) > 0
        done = np.zeros(len(g), bool)
        # Rigid override: candidates are output voxels within override_mm of a POSED bone voxel (a dense tree, so thin shells
        # such as the skull are covered; the smooth field's source is approximate after a large rotation). Each takes the exact
        # rigid-inverse source if that source lies in the bone's (dilated) mask.
        if getattr(self, "_bid_near", None) is None:
            lab = self.bone_id
            dist, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
            self._bid_near = np.where(dist <= 3.0, lab[tuple(ind)], 0).astype(lab.dtype)
            del dist, ind
        # nearest posed points of several different bones may compete near a joint (skull base / C1); try the owners in order of distance
        d, j = fld["bone_tree"].query(g, k=4, workers=-1, distance_upper_bound=override_mm)
        same = {}
        for k_, (Rb_, tb_, _, _) in fld["bones"].items():     # bones sharing one rigid motion (skull and brain) are one body
            same[k_] = [j_ for j_, (R2, t2, _, _) in fld["bones"].items() if np.abs(R2 - Rb_).max() < 1e-6 and np.abs(t2 - tb_).max() < 1e-6]
        for r in range(4):
            cand = np.flatnonzero(np.isfinite(d[:, r]) & ~done)
            if not len(cand):
                break
            owner = fld["bone_owner"][j[cand, r]]
            for k_ in np.unique(owner):
                sel = cand[owner == k_]
                Rb, tb = fld["bones"][k_][:2]
                xb = (g[sel] - tb) @ Rb                      # exact rigid inverse: x = Rb^T (y - tb)
                ok = np.isin(self.sample(self._bid_near, xb, order=0, cval=0), same[k_])
                src[sel[ok]] = xb[ok]
                done[sel[ok]] = True
        # the smooth field is only approximate for large rotations: any voxel it maps onto source bone that is NOT the exact
        # rigid image of a posed bone would draw a second, ghost copy of that bone (e.g. extra skulls in a forward bend).
        # Bones come only from the rigid override; everywhere else sample the nearest soft tissue instead.
        self._dbg = {"near": near.copy(), "done": done.copy()}
        ghost = near & ~done
        if ghost.any():
            if getattr(self, "_soft_idx", None) is None:
                self._soft_idx = ndi.distance_transform_edt(self.bone_near, return_indices=True)[1]
            vi = np.clip(np.rint(self.mm_to_vox(src[ghost])).astype(int), 0, np.array(self.bone_near.shape) - 1)
            nv = self._soft_idx[:, vi[:, 0], vi[:, 1], vi[:, 2]].T.astype(float)
            src[ghost] = self.vox_to_mm(nv)
        return src

    # ---------------------------------------------------------------- sampling and rendering
    def sample(self, vol, p_mm, order=1, cval=-1000):
        v = self.mm_to_vox(p_mm).T
        return ndi.map_coordinates(vol, v, order=order, mode="constant", cval=cval)

    def plane(self, fixed_axis, pos, hor_axis, ver_axis, center, half_h, half_v, step=1.5):
        """Points of an axis-aligned reformat plane: horizontal image axis hor_axis, vertical ver_axis (both MHR axes)."""
        hs, vs = np.arange(-half_h, half_h + step, step), np.arange(-half_v, half_v + step, step)
        H, V = np.meshgrid(hs, vs, indexing="xy")
        P = np.zeros(H.shape + (3,))
        P[..., fixed_axis] = pos
        P[..., hor_axis] = H + center[hor_axis]
        P[..., ver_axis] = V + center[ver_axis]
        return P.reshape(-1, 3), H.shape

    def warped_volume(self, fld, step=2.0, region=None, soft_bone=False, extras=None):
        """Warped CT (HU) and warped bone mask on a regular output grid, for projections.
        region: (min_xyz, max_xyz) in mm to crop the output grid.
        extras: {name: volume on the CT grid} (e.g. PET SUV, aligned MRI) warped with the same field; results in self.extra_out[name]."""
        lo = self.vox_to_mm(np.array([0, 0, 0.0])); hi = self.vox_to_mm(np.array(self.hu.shape, float) - 1)
        mn, mx = np.minimum(lo, hi), np.maximum(lo, hi)
        if region is not None:                 # explicit region wins, even beyond the CT box: a bent-forward head leaves the scanned volume
            mn, mx = np.asarray(region[0], float), np.asarray(region[1], float)
        axes = [np.arange(mn[a], mx[a] + step, step) for a in range(3)]
        shape = tuple(len(a) for a in axes)
        hu = np.empty(shape, np.float32); bone = np.empty(shape, np.float32)
        bone_f = self.bone.astype(np.float32)                 # converted once, not per slab
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        yz = np.stack([yy, zz], -1).reshape(-1, 2)
        self.extra_out = {k: np.zeros(shape, np.float32) for k in (extras or {})}
        for i0 in range(0, shape[0], 8):                      # slabs along x: peak memory stays small however large the region
            xs = axes[0][i0:i0 + 8]
            G = np.concatenate([np.concatenate([np.full((len(yz), 1), x), yz], 1) for x in xs])
            src = self.source_positions(fld, G)
            hu[i0:i0 + len(xs)] = self.sample(self.hu_c, src).astype(np.float32).reshape((len(xs),) + shape[1:])
            bone[i0:i0 + len(xs)] = self.sample(bone_f, src, cval=0).reshape((len(xs),) + shape[1:])
            for k, v in (extras or {}).items():
                self.extra_out[k][i0:i0 + len(xs)] = self.sample(v, src, cval=0).reshape((len(xs),) + shape[1:])
        if not soft_bone:
            bone = bone > 0.4                  # binary mask: thin cortex aliases (moire) when step is coarser than the CT voxel
        return hu, bone, axes, step

