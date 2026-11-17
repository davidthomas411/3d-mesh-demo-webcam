"""Cinematic volume rendering: Monte Carlo volumetric path tracing with global illumination (CPU, numba).

The same family of method as Siemens' Cinematic Rendering (Comaniciu et al. 2016): delta-tracked free-flight sampling through a
density/colour transfer function, a hybrid of volumetric phase scattering and gradient-based surface shading (Kroes et al. 2012),
next-event estimation with ratio-tracked shadows toward soft key / fill / rim lights, multiple scattering with Russian roulette,
film-style tone mapping. Not their code and not their lighting probes: it is our own implementation.

All lengths are in voxels inside the kernel; callers pass millimetres.
"""
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------------------------------- transfer functions
def _interp(x, xs, ys):
    return np.interp(x, xs, ys)


def tf_ct(hu, suv=None, mri=None, mri_scale=1.0, skin=1.0):
    """CT HU -> (sigma_t per mm, albedo rgb, emission rgb). Soft tissue stays translucent so organs read through the skin;
    bone is dense ivory; contrast-filled vessels warm red. Optional PET SUV (emission) and MRI (replaces CT where present)."""
    sig = np.zeros(hu.shape, np.float32)
    col = np.zeros(hu.shape + (3,), np.float32)
    # key points: HU, sigma (1/mm), colour
    pts = [
        (-1000, 0.0000, (0.50, 0.40, 0.40)),
        (-700, 0.0002, (0.70, 0.45, 0.45)),     # lung: faint pink haze
        (-450, 0.0006, (0.70, 0.45, 0.42)),
        (-250, 0.0020 * skin, (0.85, 0.62, 0.50)),   # skin / subcutaneous interface
        (-100, 0.0030, (0.80, 0.62, 0.42)),     # fat: pale amber, very translucent
        (-20, 0.0030, (0.78, 0.60, 0.42)),
        (10, 0.0075, (0.52, 0.17, 0.15)),       # soft tissue / organs: deep red-brown
        (70, 0.0105, (0.58, 0.16, 0.12)),
        (140, 0.0240, (0.80, 0.22, 0.10)),      # contrast-enhanced blood: orange-red
        (260, 0.0450, (0.90, 0.50, 0.30)),
        (330, 0.1500, (0.95, 0.90, 0.80)),      # cancellous bone
        (700, 0.3000, (0.98, 0.95, 0.86)),      # cortical bone: ivory
        (1500, 0.4000, (1.00, 0.99, 0.96)),
        (3000, 0.5000, (0.80, 0.82, 0.86)),     # metal / implant
    ]
    xs = [p[0] for p in pts]
    sig = _interp(hu, xs, [p[1] for p in pts]).astype(np.float32)
    for k in range(3):
        col[..., k] = _interp(hu, xs, [p[2][k] for p in pts])
    emis = np.zeros(hu.shape + (3,), np.float32)
    if suv is not None:                                   # FDG uptake: glowing hot colours where SUV is high
        t = np.clip((suv - 2.0) / 6.0, 0, 1)
        hot = np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], -1)
        mix = np.clip((suv - 1.5) / 2.0, 0, 1)[..., None]
        col = col * (1 - 0.8 * mix) + hot * 0.8 * mix
        emis = hot * (np.clip((suv - 2.5) / 5.0, 0, 1) ** 1.2)[..., None] * 1.6
        sig = np.maximum(sig, 0.03 * mix[..., 0] * (hu > -400))
    if mri is not None:                                   # MRI intensity (0..1 of p99): grey-blue translucent
        m = np.clip(mri / mri_scale, 0, 1)
        has = mri > 0.01 * mri_scale
        sig = np.where(has, 0.015 + 0.075 * m ** 1.2, sig)
        for k, v in enumerate((0.78, 0.84, 1.0)):
            col[..., k] = np.where(has, 0.25 + 0.75 * m * v, col[..., k])
    return sig, col, emis


def tf_mri(mri, p99, bone=None, organ=None, body=None):
    """MRI has no HU, so intensity alone cannot separate tissues. Intensity drives a translucent haze and modulates organ density; WHICH voxels are bone
    or organ (and their colour) comes from the TotalSegmentator labels warped with the same field (bone, organ: 0..1 masks)."""
    m = np.clip(mri / p99, 0, 1)
    sig = (0.0015 + 0.006 * m ** 1.3).astype(np.float32)
    col = np.stack([0.55 + 0.35 * m, 0.52 + 0.33 * m, 0.50 + 0.40 * m], -1).astype(np.float32)
    sig[mri < 0.03 * p99] = 0
    if organ is not None:
        w = np.clip(organ, 0, 1)
        sig = sig * (1 - w) + w * (0.010 + 0.014 * np.sqrt(m))
        for k, v in enumerate((0.62, 0.20, 0.17)):
            col[..., k] = col[..., k] * (1 - w) + w * (v * (0.7 + 0.5 * m))
    if bone is not None:
        w = np.clip(bone, 0, 1)
        sig = sig * (1 - w) + w * 0.30
        for k, v in enumerate((0.96, 0.92, 0.82)):
            col[..., k] = col[..., k] * (1 - w) + w * v
    return sig, col, np.zeros(mri.shape + (3,), np.float32)


def normals_from(vol, sigma_vox=1.2):
    """Unit surface normals (pointing out of the dense side) and gradient magnitude, from a smoothed copy of the scalar."""
    import scipy.ndimage as ndi
    s = ndi.gaussian_filter(vol.astype(np.float32), sigma_vox)
    g = np.stack(np.gradient(s), -1)
    gm = np.linalg.norm(g, axis=-1)
    n = -g / np.maximum(gm[..., None], 1e-6)
    return n.astype(np.float32), gm.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- kernel
@njit(cache=True, fastmath=True, inline="always")
def _rnd(st, i):
    x = st[i]
    x ^= x >> np.uint64(12); x ^= x << np.uint64(25); x ^= x >> np.uint64(27)
    st[i] = x
    return ((x * np.uint64(2685821657736338717)) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, fastmath=True, inline="always")
def _tri(a, x, y, z):
    nx, ny, nz = a.shape
    if x < 0 or y < 0 or z < 0 or x > nx - 1.001 or y > ny - 1.001 or z > nz - 1.001:
        return 0.0
    i, j, k = int(x), int(y), int(z)
    fx, fy, fz = x - i, y - j, z - k
    c00 = a[i, j, k] * (1 - fx) + a[i + 1, j, k] * fx
    c10 = a[i, j + 1, k] * (1 - fx) + a[i + 1, j + 1, k] * fx
    c01 = a[i, j, k + 1] * (1 - fx) + a[i + 1, j, k + 1] * fx
    c11 = a[i, j + 1, k + 1] * (1 - fx) + a[i + 1, j + 1, k + 1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


@njit(cache=True, fastmath=True, inline="always")
def _box(o, d, hi):
    t0, t1 = 0.0, 1e30
    for a in range(3):
        if abs(d[a]) < 1e-9:
            if o[a] < 0 or o[a] > hi[a]:
                return 1.0, -1.0
        else:
            ta, tb = (0.0 - o[a]) / d[a], (hi[a] - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
    return t0, t1


@njit(cache=True, fastmath=True)
def _shadow(sig, smax, p, d, hi, st, i):
    """Ratio-tracked transmittance from p along d to the volume boundary."""
    t0, t1 = _box(p, d, hi)
    if t1 <= 0:
        return 1.0
    t = 0.0
    T = 1.0
    while True:
        t += -np.log(1.0 - _rnd(st, i)) / smax
        if t >= t1:
            return T
        x, y, z = p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t
        s = _tri(sig, x, y, z)
        T *= 1.0 - s / smax
        if T < 0.02:
            return 0.0


@njit(cache=True, fastmath=True)
def _trace(sig, col, emis, nrm, gmag, smax, ox, oy, oz, dx, dy, dz, hi, lights, nl, st, i, maxb, g0, pbr_scale, spec_k):
    o = np.array([ox, oy, oz]); d = np.array([dx, dy, dz])
    thr = np.ones(3); L = np.zeros(3)
    for bounce in range(maxb):
        t0, t1 = _box(o, d, hi)
        if t1 <= t0:
            break
        t = max(t0, 0.0)
        hit = False
        while True:
            t += -np.log(1.0 - _rnd(st, i)) / smax
            if t >= t1:
                break
            x, y, z = o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t
            s = _tri(sig, x, y, z)
            if _rnd(st, i) < s / smax:
                hit = True
                break
        if not hit:
            # environment: soft top-lit dome plus a faint floor bounce (ambient light that gives the soft occlusion look)
            if bounce == 0:
                v = 0.5 - 0.5 * d[1]
                bgc = 0.002 + 0.016 * (1.0 - abs(v - 0.5) * 1.6)
                L += thr * np.array([bgc * 0.8, bgc * 0.95, bgc * 1.25])
                break
            up = -d[1]
            e = 0.10 + 0.14 * max(up, 0.0) + 0.04 * max(-up, 0.0)
            L += thr * np.array([e * 0.85, e * 0.95, e * 1.15])
            break
        ix, iy, iz = int(x + 0.5), int(y + 0.5), int(z + 0.5)
        nx_, ny_, nz_ = sig.shape
        ix, iy, iz = min(max(ix, 0), nx_ - 1), min(max(iy, 0), ny_ - 1), min(max(iz, 0), nz_ - 1)
        c = col[ix, iy, iz]
        L += thr * emis[ix, iy, iz]
        gm = gmag[ix, iy, iz]
        ps = gm / (gm + g0) * pbr_scale * min(1.0, s * 14.0)   # surface-like where the gradient is strong AND the material is dense (bone, vessels)
        p = np.array([x, y, z])
        surf = _rnd(st, i) < ps
        if surf:
            n = nrm[ix, iy, iz].astype(np.float64)
            if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0:
                n = -n
        # next-event estimation toward one soft light
        li = int(_rnd(st, i) * nl) % nl
        ld = np.array([lights[li, 0], lights[li, 1], lights[li, 2]])
        # soft light: jitter direction inside a cone
        r1, r2 = _rnd(st, i), _rnd(st, i)
        cone = lights[li, 6]
        a1 = np.sqrt(r1) * cone; ph = 6.2831853 * r2
        ax = np.cross(ld, np.array([0.0, 1.0, 0.0]) if abs(ld[1]) < 0.9 else np.array([1.0, 0.0, 0.0])); ax /= np.linalg.norm(ax)
        bx = np.cross(ld, ax)
        wl = ld + (ax * np.cos(ph) + bx * np.sin(ph)) * a1
        wl /= np.linalg.norm(wl)
        Ts = _shadow(sig, smax, p, wl, hi, st, i)
        if Ts > 0:
            if surf:
                cs = n[0] * wl[0] + n[1] * wl[1] + n[2] * wl[2]
                if cs > 0:
                    h = wl - d; h /= max(np.linalg.norm(h), 1e-9)
                    nh = max(n[0] * h[0] + n[1] * h[1] + n[2] * h[2], 0.0)
                    f_spec = spec_k * (nh ** 40.0) * (48.0 / 25.1327)
                    for k in range(3):
                        L[k] += thr[k] * Ts * lights[li, 3 + k] * nl * cs * (c[k] / 3.14159265 + f_spec)
            else:
                cosl = d[0] * wl[0] + d[1] * wl[1] + d[2] * wl[2]
                gg = 0.35
                ph_ = (1 - gg * gg) / (4 * 3.14159265 * (1 + gg * gg - 2 * gg * cosl) ** 1.5)
                for k in range(3):
                    L[k] += thr[k] * Ts * lights[li, 3 + k] * nl * ph_ * c[k]
        # scatter
        for k in range(3):
            thr[k] *= c[k]
        if surf:
            # cosine-weighted hemisphere around n (diffuse bounce)
            r1, r2 = _rnd(st, i), _rnd(st, i)
            rr = np.sqrt(r1); ph = 6.2831853 * r2
            u = np.cross(n, np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])); u /= np.linalg.norm(u)
            v = np.cross(n, u)
            nd = u * (rr * np.cos(ph)) + v * (rr * np.sin(ph)) + n * np.sqrt(max(1 - r1, 0.0))
        else:
            gg = 0.35
            r1, r2 = _rnd(st, i), _rnd(st, i)
            cth = (1 + gg * gg - ((1 - gg * gg) / (1 - gg + 2 * gg * r1)) ** 2) / (2 * gg)
            sth = np.sqrt(max(1 - cth * cth, 0.0)); ph = 6.2831853 * r2
            u = np.cross(d, np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])); u /= np.linalg.norm(u)
            v = np.cross(d, u)
            nd = u * (sth * np.cos(ph)) + v * (sth * np.sin(ph)) + d * cth
        nd /= np.linalg.norm(nd)
        o = p + nd * 0.01 if not surf else p + n * 0.6
        d = nd
        if bounce >= 3:
            q = min(max(thr[0], thr[1], thr[2]), 0.95)
            if _rnd(st, i) > q:
                break
            thr /= q
    return L[0], L[1], L[2]


@njit(parallel=True, cache=True, fastmath=True)
def _render(sig, col, emis, nrm, gmag, smax, cam, right, up, fwd, W, H, fovs, persp, spp, lights, seed, maxb, g0, pbr_scale, spec_k, hi):
    nl = lights.shape[0]
    img = np.zeros((H, W, 3))
    st = np.empty(W * H, np.uint64)
    for q in range(W * H):
        st[q] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(q + 1) * np.uint64(0xBF58476D1CE4E5B9)
    for y in prange(H):
        for x in range(W):
            i = y * W + x
            acc = np.zeros(3)
            for _ in range(8):
                _rnd(st, i)
            for s in range(spp):
                u = ((x + _rnd(st, i)) / W - 0.5) * 2.0
                v = (0.5 - (y + _rnd(st, i)) / H) * 2.0
                if persp:
                    d = fwd + right * (u * fovs * W / H) + up * (v * fovs)
                    d = d / np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                    o = cam
                else:
                    d = fwd
                    o = cam + right * (u * fovs * W / H) + up * (v * fovs)
                r, g, b = _trace(sig, col, emis, nrm, gmag, smax, o[0], o[1], o[2], d[0], d[1], d[2], hi, lights, nl, st, i, maxb, g0, pbr_scale, spec_k)
                acc[0] += r; acc[1] += g; acc[2] += b
            img[y, x, 0] = acc[0] / spp; img[y, x, 1] = acc[1] / spp; img[y, x, 2] = acc[2] / spp
    return img


# ---------------------------------------------------------------------------------------------------- python API
def default_lights():
    """key (upper left front, warm, large soft), fill (right, cool), rim (behind-above, white). Directions point TOWARD the light, in volume axes
    (x right, y down, z toward the camera for a front view where the person faces -z)."""
    def L(d, rgb, cone, k):
        d = np.asarray(d, float); d /= np.linalg.norm(d)
        return [*d, *(np.asarray(rgb) * k), cone]
    return np.array([L([-0.55, -0.60, -0.65], (1.0, 0.94, 0.86), 0.22, 4.2),
                     L([0.80, -0.20, -0.45], (0.72, 0.82, 1.0), 0.40, 1.3),
                     L([0.10, -0.55, 0.85], (1.0, 1.0, 1.0), 0.18, 2.4)], np.float64)


def tonemap(img, exposure=1.0, sat=1.08):
    x = img * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14           # ACES filmic fit (Narkowicz)
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    l = y.mean(-1, keepdims=True)
    y = np.clip(l + (y - l) * sat, 0, 1)
    return y ** (1 / 2.2)


def render(sig, col, emis, nrm, gmag, step_mm, W=480, H=720, spp=64, azimuth_deg=0.0, elevation_deg=0.0, fov_deg=14.0, dist_mm=2200.0,
           target=None, half_height_mm=None, perspective=True, seed=1, max_bounces=5, lights=None, pbr=0.9, spec=0.35, exposure=1.0, mirror_x=False):
    """Path-trace the volume. Volume axes: x person-left, y DOWN (head at small y), front of the person toward -z. The camera sits in front by default."""
    nx, ny, nz = sig.shape
    hi = np.array([nx - 1.0, ny - 1.0, nz - 1.0])
    target = (np.array(sig.shape, float) / 2.0) if target is None else np.asarray(target, float) / step_mm
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    fwd = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])        # camera looks along +z at az=0 (front is -z side)
    fwd = fwd / np.linalg.norm(fwd)
    cam = target - fwd * (dist_mm / step_mm)
    worldup = np.array([0.0, -1.0, 0.0])
    right = np.cross(fwd, worldup); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    if mirror_x:
        right = -right
    hh = (half_height_mm or (0.5 * ny * step_mm)) / step_mm
    fovs = np.tan(np.radians(fov_deg) / 2) if perspective else hh
    if perspective:
        fovs = hh / (dist_mm / step_mm)
    lights = default_lights() if lights is None else lights
    # lights are given in camera-independent volume axes; rotate with the camera azimuth so the look stays constant
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    lights = lights.copy(); lights[:, :3] = lights[:, :3] @ Ry.T
    smax = float(sig.max()) * step_mm
    s_vox = (sig * step_mm).astype(np.float32)
    g0 = float(np.percentile(gmag[gmag > 0], 70)) if (gmag > 0).any() else 1.0
    img = _render(s_vox, col.astype(np.float32), emis.astype(np.float32), nrm, gmag, smax, cam, right, up, fwd, W, H, fovs, perspective, spp, lights, seed,
                  max_bounces, g0, pbr, spec, hi)
    return img


def denoise(img, strength=0.6):
    """Light, edge-preserving denoise on the HDR (tone-mapped) image with an a-trous guided filter; keeps bone edges."""
    from skimage.restoration import denoise_nl_means, estimate_sigma
    sg = float(np.mean(estimate_sigma(img, channel_axis=-1)))
    return denoise_nl_means(img, patch_size=5, patch_distance=6, h=strength * sg, sigma=sg, fast_mode=True, channel_axis=-1)
