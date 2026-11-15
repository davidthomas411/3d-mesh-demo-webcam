"""Deform the atlas subject's REAL CT with the anatomy rig's motion, then render radiographs and reformats.

What this is: a dense displacement field built from how every structure in the rig moves (bones: exact rigid
motion in their own CT frame; soft tissue: the skinned motion), interpolated by inverse-distance weighting and
applied by backward warping. It gives a deformed CT with a known voxel-to-voxel correspondence to the original.
What it is NOT: a physical simulation. Tissue far from any modelled structure is interpolated; sliding,
breathing and gas are not modelled; arms and couch are removed.

Frames: everything in the CT's own frame, millimetres, MHR axes (x = patient left, y = down, z = back).
"""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from anatomy import atlas_build as ab
from anatomy import rig
from anatomy.runtime import Atlas

SKIP_LANDMARKS = ("humerus_", "femur_")       # arms/legs are cropped; their CT pose differs from the atlas T-pose


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
        bid = np.where(np.isin(self.label, bone_ids), self.label, 0).astype(np.uint8)
        self.bone_id = ndi.grey_dilation(bid, size=3)                       # bone identity per voxel (for exact rigid motion)
        self.bone_near = ndi.binary_dilation(self.bone, iterations=3)       # candidates for the rigid override
        # ---- remove couch and arms
        self.keep = self._body_mask() & self._lateral_mask(margin_mm)
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
    def landmarks(self, posed, skin_posed, structures=True):
        """-> X (source, CT frame mm), Y (posed, CT frame mm), and per-bone data for the exact rigid override."""
        s_w, R_w, _ = self.W
        X, Y, bones = [], [], {}
        for n in (self.names if structures else []):
            if n.startswith(SKIP_LANDMARKS):
                continue
            xs = self.v_ct[n][::2]
            p = posed[n][::2]
            if ab.is_bone(n):
                s, R, t = self.T[n]
                ys = ((p - t) @ R) / s                      # posed vertex expressed in the bone's own CT frame (exact, rigid)
                _, Rb, tb = ab.umeyama(xs, ys, with_scale=False)   # rigid motion of the bone in the CT frame: y = Rb x + tb
                bones[self.label_id[n]] = (Rb, tb, xs, ys)
            else:
                rest = self.by[n]["pos"][::2]
                ys = xs + ((p - rest) @ R_w) / s_w          # skinned motion, rotated into the CT frame
            X.append(xs); Y.append(ys)
        # the outer body follows the posed MHR skin
        du = (skin_posed[self.skin_nn] - self.mhr_rest[self.skin_nn]) @ R_w / s_w
        X.append(self.skin_ct); Y.append(self.skin_ct + du)
        return np.vstack(X), np.vstack(Y), bones

    def field(self, posed, skin_posed, grid_mm=6.0, k=32, sigma_mm=10.0, structures=True):
        """Coarse backward displacement field u_hat(g) for output (posed) positions g, plus the exact bone motions."""
        X, Y, bones = self.landmarks(posed, skin_posed, structures)
        U = Y - X
        tree = cKDTree(Y)
        lo = self.vox_to_mm(np.array([0, 0, 0.0])); hi = self.vox_to_mm(np.array(self.hu.shape, float) - 1)
        mn, mx = np.minimum(lo, hi), np.maximum(lo, hi)
        axes = [np.arange(mn[a], mx[a] + grid_mm, grid_mm) for a in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        d, i = tree.query(G, k=k, workers=-1)
        w = np.exp(-(d ** 2 - d[:, :1] ** 2) / (2 * sigma_mm ** 2))        # Gaussian, relative to the nearest landmark
        uh = (w[..., None] * U[i]).sum(1) / w.sum(1, keepdims=True)
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
        d, j = fld["bone_tree"].query(g, k=1, workers=-1, distance_upper_bound=override_mm)
        idx = np.flatnonzero(np.isfinite(d))
        if len(idx):
            owner = fld["bone_owner"][j[idx]]
            for k_ in np.unique(owner):
                sel = idx[owner == k_]
                Rb, tb = fld["bones"][k_][:2]
                xb = (g[sel] - tb) @ Rb                      # exact rigid inverse: x = Rb^T (y - tb)
                ok = self.sample(self._bid_near, xb, order=0, cval=0) == k_
                src[sel[ok]] = xb[ok]
                done[sel[ok]] = True
        # the smooth field is only approximate for large rotations: any voxel it maps onto source bone that is NOT the exact
        # rigid image of a posed bone would draw a second, ghost copy of that bone (e.g. extra skulls in a forward bend).
        # Bones come only from the rigid override; everywhere else sample the nearest soft tissue instead.
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

    def warped_volume(self, fld, step=2.0, region=None, soft_bone=False):
        """Warped CT (HU) and warped bone mask on a regular output grid, for projections.
        region: (min_xyz, max_xyz) in mm to crop the output grid."""
        lo = self.vox_to_mm(np.array([0, 0, 0.0])); hi = self.vox_to_mm(np.array(self.hu.shape, float) - 1)
        mn, mx = np.minimum(lo, hi), np.maximum(lo, hi)
        if region is not None:
            mn, mx = np.maximum(mn, region[0]), np.minimum(mx, region[1])
        axes = [np.arange(mn[a], mx[a] + step, step) for a in range(3)]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        src = self.source_positions(fld, G)
        shape = tuple(len(a) for a in axes)
        hu = self.sample(self.hu_c, src).astype(np.float32).reshape(shape)
        bone = self.sample(self.bone.astype(np.float32), src, cval=0).reshape(shape)
        if not soft_bone:
            bone = bone > 0.4                  # binary mask: thin cortex aliases (moire) when step is coarser than the CT voxel
        return hu, bone, axes, step

