// Anatomy layer: TotalSegmentator-class structures driven by a live MHR skin mesh.
// Port of anatomy/runtime.py (v2 rigged atlas). ES module, three.js is passed in.
//
// Biomechanical model:
//  * rigid SEGMENTS (pelvis, lumbar, lower/upper thorax, head, upper arms, thighs) get a
//    rest->live rigid transform from their supporting skin vertices (Horn);
//  * every vertebra is a rigid body: quaternion-blended rotation, forward-kinematic positions
//    (spacing preserved) plus a smoothed skin-driven offset for low-frequency placement;
//  * ribs / sternum / cartilages / clavicles / scapulae ride on the vertebrae they articulate
//    with; pelvis, skull, limb bones follow their segment; ALL BONES ARE RIGID;
//  * soft tissue is skinned between the spine-chain frames that bracket its height.
// ILLUSTRATIVE atlas anatomy, not patient-specific: present it as such (DISCLAIMER).

export const DISCLAIMER = "Atlas-based illustrative anatomy. Not patient-specific. Not for clinical use.";

export async function loadAtlas(jsonUrl, binUrl) {
  const [meta, buf] = await Promise.all([
    fetch(jsonUrl).then(r => { if (!r.ok) throw new Error("atlas json " + r.status); return r.json(); }),
    fetch(binUrl).then(r => { if (!r.ok) throw new Error("atlas bin " + r.status); return r.arrayBuffer(); }),
  ]);
  return parseAtlas(meta, buf);
}

// meta: parsed atlas.json, buf: ArrayBuffer of atlas.bin (usable from Node with fs buffers)
export function parseAtlas(meta, buf) {
  if (meta.version !== 2) throw new Error("atlas v2 required");
  const segments = meta.segments.map(g => ({
    name: g.name,
    idx: Uint32Array.from(new Uint16Array(buf, g.idx, g.n)),
    rest: new Float32Array(buf.slice(g.rest, g.rest + g.n * 12)),
  }));
  const structures = meta.structures.map(e => {
    const n = e.n_verts;
    const s = {
      ...e,
      pos: new Float32Array(buf.slice(e.pos, e.pos + n * 12)),
      faces: e.idx_type === "u16" ? Uint32Array.from(new Uint16Array(buf, e.idx, e.n_tris * 3))
                                  : new Uint32Array(buf.slice(e.idx, e.idx + e.n_tris * 12)),
    };
    if (e.bind === "soft") {
      s.infIdx = new Uint8Array(buf.slice(e.inf_idx, e.inf_idx + n * 3));
      s.infW = new Float32Array(buf.slice(e.inf_w, e.inf_w + n * 12));
    }
    return s;
  });
  const byName = new Map(structures.map(s => [s.name, s]));
  if (meta.mhr) {                                              // rest_pos / rest_rot are keyed by joint index
    for (const k of ["rest_pos", "rest_rot"]) { const o = []; for (const j in meta.mhr[k]) o[Number(j)] = meta.mhr[k][j]; meta.mhr[k] = o; }
  }
  prepareCartilage(structures, byName);
  return { meta, segments, structures, byName, chain: meta.chain || [], joints: meta.joints || [], mhr: meta.mhr || null };
}

// Costal cartilage is a flexible link: each vertex follows the rib nearest to it at the rib end and the sternum at the sternal
// end, blended by rest distance (mirrors Atlas._cartilage in anatomy/runtime.py), so neither junction can open.
function prepareCartilage(structures, byName) {
  const c = byName.get("costal_cartilages"), st = byName.get("sternum");
  if (!c || !st) return;
  const ribs = structures.filter(s => s.name.startsWith("rib_"));
  const nearest = (p, i, pos) => {
    let best = Infinity, bj = -1;
    for (let j = 0; j < pos.length; j += 3) {
      const dx = p[i] - pos[j], dy = p[i + 1] - pos[j + 1], dz = p[i + 2] - pos[j + 2], d = dx * dx + dy * dy + dz * dz;
      if (d < best) { best = d; bj = j; }
    }
    return [Math.sqrt(best), bj];
  };
  const n = c.pos.length / 3; c.cartOwner = new Array(n); c.cartW = new Float32Array(n);
  for (let v = 0; v < n; v++) {
    let dr = Infinity, own = null;
    for (const r of ribs) { const [d] = nearest(c.pos, 3 * v, r.pos); if (d < dr) { dr = d; own = r.name; } }
    const [ds] = nearest(c.pos, 3 * v, st.pos);
    c.cartOwner[v] = own; c.cartW[v] = dr / Math.max(dr + ds, 1e-9);
  }
}

// Largest-eigenvalue eigenvector of a symmetric 4x4 by cyclic Jacobi (robust, tiny).
function topEigen4(N) {
  const a = N.map(r => r.slice());
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 24; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 4; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 4; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

// Weighted rigid fit (Horn 1987): returns row-major R (9), rest centroid cs (3), live centroid cd (3).
export function weightedRigid(sRest, sIdx, sW, live, out) {
  // sW may be null for uniform weights
  const n = sIdx.length;
  let wsum = 0;
  const wAt = i => (sW ? sW[i] : 1);
  for (let i = 0; i < n; i++) wsum += wAt(i);
  const cs = [0, 0, 0], cd = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = wAt(i) / wsum, j = sIdx[i] * 3;
    for (let k = 0; k < 3; k++) { cs[k] += w * sRest[i * 3 + k]; cd[k] += w * live[j + k]; }
  }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // S[a][b] = sum w (src_a - cs_a)(dst_b - cd_b)
  for (let i = 0; i < n; i++) {
    const w = wAt(i) / wsum, j = sIdx[i] * 3;
    for (let a = 0; a < 3; a++) {
      const sa = (sRest[i * 3 + a] - cs[a]) * w;
      for (let b = 0; b < 3; b++) S[a][b] += sa * (live[j + b] - cd[b]);
    }
  }
  const [Sxx, Sxy, Sxz] = S[0], [Syx, Syy, Syz] = S[1], [Szx, Szy, Szz] = S[2];
  const q = topEigen4([
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ]);
  const [w, x, y, z] = q;
  const R = out || new Float64Array(9);
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
  return { R, q, cs, cd };
}

function quatToR(q, out) {
  const [w, x, y, z] = q, R = out || new Float64Array(9);
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
  return R;
}
const mapPoint = (f, p) => {                // x -> s R (x - c) + cp; s is the frame's uniform scale (absent = 1)
  const s = f.s || 1, x = s * (p[0] - f.c[0]), y = s * (p[1] - f.c[1]), z = s * (p[2] - f.c[2]), R = f.R;
  return [R[0] * x + R[1] * y + R[2] * z + f.cp[0], R[3] * x + R[4] * y + R[5] * z + f.cp[1], R[6] * x + R[7] * y + R[8] * z + f.cp[2]];
};
function blendQuat(frames) {         // [[frame, w], ...] -> normalised quaternion, sign-aligned
  const ref = frames[0][0].q, q = [0, 0, 0, 0];
  for (const [f, w] of frames) {
    const d = f.q[0] * ref[0] + f.q[1] * ref[1] + f.q[2] * ref[2] + f.q[3] * ref[3], sg = d >= 0 ? 1 : -1;
    for (let k = 0; k < 4; k++) q[k] += w * sg * f.q[k];
  }
  const n = Math.hypot(...q);
  return q.map(v => v / n);
}
// One rigid motion for a bone from one or more frames: quaternion-blended rotation, with the
// centroid carried by the weighted frame images (exact for a single frame).
function rigidFromFrames(frames, centroid) {
  if (frames.length === 1) { const f = frames[0][0]; return { R: f.R, cp: mapPoint(f, centroid), s: f.s || 1 }; }
  const R = quatToR(blendQuat(frames)), cp = [0, 0, 0];
  let s = 0, ws = 0;
  for (const [f, w] of frames) { const m = mapPoint(f, centroid); for (let k = 0; k < 3; k++) cp[k] += w * m[k]; s += w * (f.s || 1); ws += w; }
  return { R, cp, s: s / ws };
}
function blendFrame(centroid, wf) {
  const q = blendQuat(wf), cp = [0, 0, 0];
  let s = 0, ws = 0;
  for (const [f, w] of wf) { const m = mapPoint(f, centroid); for (let k = 0; k < 3; k++) cp[k] += w * m[k]; s += w * (f.s || 1); ws += w; }
  return { q, R: quatToR(q), c: centroid, cp, s: s / ws };
}

// ---- small rotation toolkit (row-major 3x3 in Float64Array(9)); mirrors anatomy/runtime.py
const mul3 = (A, B) => {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
};
const tr3 = A => Float64Array.of(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);
function matToQuat(R) {                       // same branch structure as the numpy reference
  const t = R[0] + R[4] + R[8];
  let q;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; q = [0.25 * s, (R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s]; }
  else if (R[0] > R[4] && R[0] > R[8]) { const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2; q = [(R[7] - R[5]) / s, 0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s]; }
  else if (R[4] > R[8]) { const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2; q = [(R[2] - R[6]) / s, (R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s]; }
  else { const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2; q = [(R[3] - R[1]) / s, (R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s]; }
  const n = Math.hypot(...q);
  return q.map(v => v / n);
}
function rodrigues(axis, ang) {
  const n = Math.hypot(...axis);
  if (n < 1e-12 || Math.abs(ang) < 1e-14) return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  const [x, y, z] = axis.map(v => v / n), c = Math.cos(ang), s = Math.sin(ang), k = 1 - c;
  return Float64Array.of(c + x * x * k, x * y * k - z * s, x * z * k + y * s,
                         y * x * k + z * s, c + y * y * k, y * z * k - x * s,
                         z * x * k - y * s, z * y * k + x * s, c + z * z * k);
}
function matToRotvec(R) {
  let q = matToQuat(R);
  if (q[0] < 0) q = q.map(v => -v);
  const s = Math.hypot(q[1], q[2], q[3]);
  if (s < 1e-12) return [0, 0, 0];
  const ang = 2 * Math.atan2(s, q[0]);
  return [q[1] / s * ang, q[2] / s * ang, q[3] / s * ang];
}
const DEG = Math.PI / 180;
function limitRotation(R, maxDeg) {           // clamp the angle of a relative rotation (axis kept)
  const rv = matToRotvec(R), ang = Math.hypot(...rv), m = maxDeg * DEG;
  return ang > m ? rodrigues(rv, m) : R;
}
function aim(a, b, maxDeg) {                  // shortest-arc rotation taking a onto the direction of b, clamped
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  if (na < 1e-9 || nb < 1e-9) return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  const A = a.map(v => v / na), B = b.map(v => v / nb);
  const v = [A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]];
  const c = Math.max(-1, Math.min(1, A[0] * B[0] + A[1] * B[1] + A[2] * B[2])), sn = Math.hypot(...v);
  if (sn < 1e-9) return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  return rodrigues(v, Math.min(Math.atan2(sn, c), maxDeg * DEG));
}
// Scapulothoracic contact (mirrors _glide in anatomy/runtime.py): swing the scapula about its acromioclavicular pin until the mean gap of its
// contact patch to the thorax ellipsoid is back to the rest value. Fixed iteration count.
function glide(ac, R0, pJac, S, pos, pin = null) {              // pin: rest point the scapula turns about (default: the AC joint)
  if (!ac.patch) return R0;
  const c = ac.ellipsoid.center, r = ac.ellipsoid.radii, J = pin || ac.center, SR = S.R, ks = S.s || 1;   // ks: the thorax may be scaled
  let R = R0;
  for (let it = 0; it < 6; it++) {
    const my = [0, 0, 0], nb = [0, 0, 0]; let dsum = 0, dmin = Infinity;
    for (const i of ac.patch) {
      const p = [pos[3 * i] - J[0], pos[3 * i + 1] - J[1], pos[3 * i + 2] - J[2]];
      const y = [0, 1, 2].map(k => ks * (R[k * 3] * p[0] + R[k * 3 + 1] * p[1] + R[k * 3 + 2] * p[2]) + pJac[k]);
      const t = [(y[0] - S.cp[0]) / ks, (y[1] - S.cp[1]) / ks, (y[2] - S.cp[2]) / ks];
      const x = [0, 1, 2].map(k => S.c[k] + SR[k] * t[0] + SR[3 + k] * t[1] + SR[6 + k] * t[2]);       // S.c + R^T (y - S.cp)
      const dx = [x[0] - c[0], x[1] - c[1], x[2] - c[2]];
      const f = Math.hypot(dx[0] / r[0], dx[1] / r[1], dx[2] / r[2]);
      const dd = Math.hypot(...dx) * (1 - 1 / Math.max(f, 1e-9)); dsum += dd; if (dd < dmin) dmin = dd;
      const g = [dx[0] / (r[0] * r[0]), dx[1] / (r[1] * r[1]), dx[2] / (r[2] * r[2])], gn = Math.max(Math.hypot(...g), 1e-12);
      for (let k = 0; k < 3; k++) { nb[k] += (SR[k * 3] * g[0] + SR[k * 3 + 1] * g[1] + SR[k * 3 + 2] * g[2]) / gn; my[k] += y[k]; }
    }
    const n = ac.patch.length, nn = Math.max(Math.hypot(...nb), 1e-12), delta = (dsum / n - ac.gap0) - Math.max(ac.gapmin0 - dmin, 0);
    const lever = [my[0] / n - pJac[0], my[1] / n - pJac[1], my[2] / n - pJac[2]], l2 = Math.max(lever[0] ** 2 + lever[1] ** 2 + lever[2] ** 2, 1e-9);
    const u = nb.map(v => -delta * ks * v / nn);
    let w = [lever[1] * u[2] - lever[2] * u[1], lever[2] * u[0] - lever[0] * u[2], lever[0] * u[1] - lever[1] * u[0]].map(v => v / l2);
    const a = Math.hypot(...w); if (a > 0.15) w = w.map(v => v * 0.15 / a);
    const ang = 0.8 * Math.hypot(...w);
    if (ang > 1e-12) R = mul3(rodrigues(w, ang), R);
  }
  return R;
}

const mkFrame = (R, J, pJ, s = 1) => ({ q: matToQuat(R), R, c: J, cp: pJ, s });

// ---------------------------------------------------------------------------------------------
// Kinematics (no three.js): segment frames -> spine chain -> joint frames -> soft-tissue frames.
// breath: 0 (end-expiration) .. 1 (end-inspiration); depth: 1 = quiet breathing.
// Physiologic distribution of trunk motion along the vertebrae (spine = "physio"; mirrors physio_cumulative in anatomy/runtime.py).
// Per-level axial rotation (Fujimori 2012 thoracic; lumbar chosen inside Ochia 2006 / Pearcy 1984 ranges), lateral bending and flexion shares: see runtime.py for sources.
const PH_AXIAL = { L5: .8, L4: 1.6, L3: 2.0, L2: 2.0, L1: 1.8, T12: .5, T11: 1.3, T10: 2.6, T9: 2.7, T8: 2.5, T7: 2.3, T6: 1.9, T5: 1.8, T4: 1.6, T3: 1.4, T2: 1.6, T1: 1.2 };
const PH_TW = {}; for (let k = 1; k <= 12; k++) PH_TW["T" + k] = k <= 4 ? 0.8 : k <= 8 ? 1.0 : 1.3;
const PH_TWSUM = Object.values(PH_TW).reduce((a, b) => a + b, 0);
const PH_LAT = {}, PH_FLX = {};
[["L5", 3], ["L4", 6], ["L3", 10], ["L2", 10], ["L1", 10]].forEach(([n, w]) => { PH_LAT[n] = 17.35 * w / 39.0; PH_FLX[n] = 55.4 / 5; });
Object.entries(PH_TW).forEach(([n, w]) => { PH_LAT[n] = 25.5 * w / PH_TWSUM; PH_FLX[n] = 19.9 * w / PH_TWSUM; });
function physioCumulative(chain) {
  const names = chain.filter(n => (n.replace("vertebrae_", "") in PH_AXIAL)), tabs = [PH_FLX, PH_AXIAL, PH_LAT];
  const tot = tabs.map(t => names.reduce((a, n) => a + t[n.replace("vertebrae_", "")], 0)), run = [0, 0, 0], out = {};
  for (const n of names) { const k = n.replace("vertebrae_", ""); tabs.forEach((t, a) => { run[a] += t[k]; }); out[n] = run.map((v, a) => v / tot[a]); }
  return out;
}

// mhrJoints: optional { pos: Float32Array(J*3), rot: Float32Array(J*9, row-major) }, the MHR joint positions and global rotations the
// estimator predicts (atlas frame). A segment bound to a joint (atlas.mhr, tools/bind_mhr_joints.py) then takes that joint's rest->live
// ROTATION instead of a fit to its skin patch; its position still comes from the patch (rest centroid -> live centroid), which keeps
// the bones inside the body better than anchoring to the joint centres.
// bodyScale: every frame carries a uniform scale s (the person's segment size relative to the neutral body, segmentScale) applied
// about the frame's rest centre, so the skeleton is sized to the body it sits in. Mirrors Atlas.segment_transforms / frames in runtime.py.
export function computeFrames(atlas, skin, breath = 0, depth = 1, spine = "skin", supportStride = 1, mhrJoints = null, bodyScale = true) {
  const SEG = Object.fromEntries(atlas.segments.map((g, i) => [g.name, i]));
  const mhr = mhrJoints ? atlas.mhr : null;
  const T = atlas.segments.map(g => {
    const j = mhr ? mhr.segments[g.name] : undefined;
    if (j !== undefined) {
      const Rr = mhr.rest_rot[j], L = mhrJoints.rot, o = j * 9, M = new Float64Array(9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[a * 3 + b] = L[o + a * 3] * Rr[b * 3] + L[o + a * 3 + 1] * Rr[b * 3 + 1] + L[o + a * 3 + 2] * Rr[b * 3 + 2];   // R_live R_rest^T
      const q0 = matToQuat(M), n = Math.hypot(q0[0], q0[1], q0[2], q0[3]) || 1, qn = [q0[0] / n, q0[1] / n, q0[2] / n, q0[3] / n];
      const R = quatToR(qn);                                     // exactly orthonormal (the input may have been extrapolated)
      const cen = patchCentroids(g, skin);
      return { q: matToQuat(R), R, c: cen.cs, cp: cen.cd, s: bodyScale ? segmentScale(g, skin) : 1 };
    }
    let rest = g.rest, idx = g.idx;
    if (supportStride > 1) {                                     // fit each segment on every k-th support vertex: ~k times faster, sub-millimetre change
      const key = "_s" + supportStride;
      if (!g[key]) {
        const m = Math.ceil(g.idx.length / supportStride), r2 = new Float32Array(m * 3), i2 = new Uint32Array(m);
        for (let a = 0; a < m; a++) { const b = a * supportStride; i2[a] = g.idx[b]; r2[a * 3] = g.rest[b * 3]; r2[a * 3 + 1] = g.rest[b * 3 + 1]; r2[a * 3 + 2] = g.rest[b * 3 + 2]; }
        g[key] = { rest: r2, idx: i2 };
      }
      rest = g[key].rest; idx = g[key].idx;
    }
    const r = weightedRigid(rest, idx, null, skin);
    return { q: r.q, R: r.R, c: r.cs, cp: r.cd, s: bodyScale ? segmentScale(g, skin) : 1 };
  });
  if (SEG.head !== undefined) T[SEG.head] = craniumFrame(atlas, skin, T[SEG.head], bodyScale);   // head: placed and sized by the rigid cranium skin
  const chain = atlas.chain, n = chain.length, cf = [];
  if (n) {
    const q = [], c = [], sp = [], kc = [];                      // kc: size of each vertebra's level
    for (const name of chain) {
      const s = atlas.byName.get(name), wf = s.bone_w.map(([sg, w]) => [T[sg], w]);
      q.push(blendQuat(wf)); c.push(s.centroid);
      const p = [0, 0, 0]; let kk = 0;
      for (const [f, w] of wf) { const m = mapPoint(f, s.centroid); for (let k = 0; k < 3; k++) p[k] += w * m[k]; kk += w * (f.s || 1); }
      sp.push(p); kc.push(kk);
    }
    if (spine === "physio") {                                    // share the measured pelvis->thorax rotation along the vertebrae by in-vivo per-level data
      const Rp = T[SEG.pelvis].R, Rt = T[SEG.thorax_up].R, rv = matToRotvec(mul3(tr3(Rp), Rt));
      const cum = atlas._physio || (atlas._physio = physioCumulative(chain));
      for (let i = 0; i < n; i++) {
        const fr = cum[chain[i]]; if (!fr) continue;
        const w = [fr[0] * rv[0], fr[1] * rv[1], fr[2] * rv[2]];
        q[i] = matToQuat(mul3(Rp, rodrigues(w, Math.hypot(...w))));
      }
    }
    const cp = [sp[0].slice()];
    for (let i = 1; i < n; i++) {
      const d = q[i][0] * q[i - 1][0] + q[i][1] * q[i - 1][1] + q[i][2] * q[i - 1][2] + q[i][3] * q[i - 1][3];
      const sg = d >= 0 ? 1 : -1, qm = q[i - 1].map((v, k) => v + sg * q[i][k]), nn = Math.hypot(...qm);
      const R = quatToR(qm.map(v => v / nn)), dv = [c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1], c[i][2] - c[i - 1][2]];
      const km = 0.5 * (kc[i - 1] + kc[i]);
      cp.push([0, 1, 2].map(k => cp[i - 1][k] + km * (R[k * 3] * dv[0] + R[k * 3 + 1] * dv[1] + R[k * 3 + 2] * dv[2])));
    }
    // low-frequency skin anchoring: final = FK + gaussian-smooth(skin - FK), sigma = 3 vertebrae
    const ker = []; let ks = 0;
    for (let k = -6; k <= 6; k++) { const v = Math.exp(-k * k / 18); ker.push(v); ks += v; }
    for (let i = 0; i < n; i++) {
      const out = [0, 0, 0];
      for (let k = -6; k <= 6; k++) {
        const j = Math.min(n - 1, Math.max(0, i + k)), w = ker[k + 6] / ks;
        for (let m = 0; m < 3; m++) out[m] += w * (sp[j][m] - cp[j][m]);
      }
      cf.push({ q: q[i], R: quatToR(q[i]), c: c[i], cp: [cp[i][0] + out[0], cp[i][1] + out[1], cp[i][2] + out[2]], s: kc[i] });
    }
  }
  const F = jointFrames(atlas, T, cf, SEG, breath, depth, mhr ? mhrJoints : null, T[SEG.head]);
  if (mhr && mhr.limbs) {                                      // limb bones sized to this person's bone lengths
    const P = mhrJoints.pos;
    for (const bone in mhr.limbs) {
      if (!F[bone]) continue;
      const lb = mhr.limbs[bone], a = lb.joint * 3, b = lb.child * 3;
      // uniform scale: the smaller of the length ratio and the limb's girth ratio, so a long thin arm does not get a thick bone
      F[bone].s = Math.min(F[bone].s || 1, Math.min(1.25, Math.max(0.8, Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]) / lb.rest_len)));
    }
  }
  const soft = [T[SEG.pelvis], ...cf, T[SEG.head], T[SEG.uparm_l], T[SEG.uparm_r], T[SEG.thigh_l], T[SEG.thigh_r]];
  return { T, cf, soft, F, mhr: mhrJoints && atlas.mhr ? { meta: atlas.mhr, joints: mhrJoints, scale: bodyScale, cache: {}, skin } : null, jawRad: 0 };
}

// Frame of a bone that is rigid with an MHR joint ("mhr" bind): it keeps its rest place relative to the joint, turns with it and is
// scaled by the joint-to-child length. Mirrors Atlas.mhr_frame in runtime.py. null without joints.
function mhrFrame(fr, bind) {
  const m = fr.mhr;
  if (!m) return null;
  const key = bind.joint + ":" + (bind.child ?? "");
  if (m.cache[key]) return m.cache[key];
  const j = bind.joint, Rr = m.meta.rest_rot[j], L = m.joints.rot, P = m.joints.pos, o = j * 9, M = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[a * 3 + b] = L[o + a * 3] * Rr[b * 3] + L[o + a * 3 + 1] * Rr[b * 3 + 1] + L[o + a * 3 + 2] * Rr[b * 3 + 2];
  const q0 = matToQuat(M), n = Math.hypot(q0[0], q0[1], q0[2], q0[3]) || 1, R = quatToR([q0[0] / n, q0[1] / n, q0[2] / n, q0[3] / n]);
  let s = 1, ax = null, sl = 1;
  if (bind.child !== undefined && m.scale) {
    const c = bind.child * 3, a = j * 3, d = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]], len = Math.hypot(d[0], d[1], d[2]);
    s = Math.min(1.25, Math.max(0.8, len / bind.rest_len));
    if (bind.girth_idx && len > 1e-6) {
      // length along the bone from the joints, thickness from the limb's skin: a long thin limb gets a long thin bone
      const u = [d[0] / len, d[1] / len, d[2] / len], skin = m.skin; let sum = 0;
      for (const vi of bind.girth_idx) {
        const x = skin[vi * 3] - P[a], y = skin[vi * 3 + 1] - P[a + 1], z = skin[vi * 3 + 2] - P[a + 2], t = x * u[0] + y * u[1] + z * u[2];
        const rx = x - t * u[0], ry = y - t * u[1], rz = z - t * u[2];
        sum += rx * rx + ry * ry + rz * rz;
      }
      const kg = Math.min(1.3, Math.max(0.6, Math.sqrt(sum / bind.girth_idx.length) / bind.girth_rest));
      const rj = m.meta.rest_pos[j], rc = m.meta.rest_pos[bind.child], r = [rc[0] - rj[0], rc[1] - rj[1], rc[2] - rj[2]], rn = Math.hypot(r[0], r[1], r[2]);
      ax = [r[0] / rn, r[1] / rn, r[2] / rn]; sl = s; s = kg;
    }
  }
  return (m.cache[key] = { R, c: m.meta.rest_pos[j], cp: [P[j * 3], P[j * 3 + 1], P[j * 3 + 2]], s, ax, sl });
}

// Body frames {q,R,c,cp} (x' = R (x - c) + cp) for every joint-driven bone. Mirrors _joint_frames in runtime.py.
// Size of a segment's live skin patch relative to its rest patch (ratio of RMS radii about the centroids), clamped. The neutral MHR
// body is one size and the estimator predicts each person's own shape, so a rigid bone would otherwise stay atlas-sized in a head
// that is bigger or smaller. Mirrors Atlas.segment_scale in anatomy/runtime.py.
// Frame that carries the skull: the head segment's rotation, placed and sized by the skin that is rigid with the head joint
// (atlas.mhr.cranium: no neck, no jaw). Mirrors Atlas._cranium_frame in runtime.py.
function craniumFrame(atlas, skin, Th, bodyScale) {
  const cr = atlas.mhr && atlas.mhr.cranium;
  if (!cr) return Th;
  if (!cr._g) cr._g = { idx: Uint32Array.from(cr.idx), rest: Float32Array.from(cr.rest.flat()) };
  const { cs, cd } = patchCentroids(cr._g, skin);
  return { q: Th.q, R: Th.R, c: cs, cp: cd, s: bodyScale ? segmentScale(cr._g, skin) : 1 };
}
function patchCentroids(g, skin) {             // rest and live centroid of a segment's skin patch
  const r = g.rest, idx = g.idx, n = idx.length, cs = [0, 0, 0], cd = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const j = idx[i] * 3;
    cs[0] += r[i * 3]; cs[1] += r[i * 3 + 1]; cs[2] += r[i * 3 + 2];
    cd[0] += skin[j]; cd[1] += skin[j + 1]; cd[2] += skin[j + 2];
  }
  return { cs: cs.map(v => v / n), cd: cd.map(v => v / n) };
}
function segmentScale(g, skin, lo = 0.65, hi = 1.35) {
  let a = 0, b = 0;
  const r = g.rest, idx = g.idx, { cs, cd } = patchCentroids(g, skin);
  for (let i = 0; i < idx.length; i++) {
    const j = idx[i] * 3;
    const x = r[i * 3] - cs[0], y = r[i * 3 + 1] - cs[1], z = r[i * 3 + 2] - cs[2];
    const u = skin[j] - cd[0], v = skin[j + 1] - cd[1], w = skin[j + 2] - cd[2];
    a += x * x + y * y + z * z; b += u * u + v * v + w * w;
  }
  return Math.min(hi, Math.max(lo, Math.sqrt(b / Math.max(a, 1e-12))));
}

// With mhrJoints and an atlas whose arm bones are seated on the MHR skeleton (atlas.mhr.limb_seated), the humeral head rides the live
// MHR shoulder joint, the scapula is pinned there and the clavicle reaches from the sternum to the scapula (see runtime.py).
function jointFrames(atlas, T, cf, SEG, breath, depth, mhrJoints = null, headFrame = null) {
  const chain = atlas.chain, J = Object.fromEntries(atlas.joints.map(j => [j.name, j])), F = {};
  for (const j of atlas.joints) {                              // ribs: costovertebral joint + respiratory rotation
    if (j.type !== "costovertebral" || !chain.includes(j.parent)) continue;
    const fv = cf[chain.indexOf(j.parent)], ang = j.resp_deg * depth * breath * DEG * j.resp_sign;
    F[j.child] = mkFrame(mul3(fv.R, rodrigues(j.axis, ang)), j.center, mapPoint(fv, j.center), fv.s || 1);
  }
  const st = atlas.byName.get("sternum");
  if (st && st.attach_ribs) {
    const wf = st.attach_ribs.filter(([r]) => F[r]).map(([r, w]) => [F[r], w]);
    if (wf.length) F.sternum = blendFrame(st.centroid, wf);
  }
  const S = F.sternum, Rthup = T[SEG.thorax_up].R;
  for (const k of ["l", "r"]) {
    const Rarm = T[SEG["uparm_" + k]].R, Rthigh = T[SEG["thigh_" + k]].R;
    const sc = J["sc_" + k], ac = J["ac_" + k], gh = J["gh_" + k], hip = J["hip_" + k];
    const mhr = mhrJoints && atlas.mhr && atlas.mhr.limb_seated ? atlas.mhr : null;
    const lb = mhr ? mhr.limbs["humerus_" + (k === "l" ? "left" : "right")] : null;
    if (S && lb && sc && ac && gh) {
      const P = mhrJoints.pos, o = lb.joint * 3, head = [P[o], P[o + 1], P[o + 2]], ks = S.s || 1;   // live MHR shoulder joint = centre of the humeral head
      // scapula: glenoid on the humeral head; a third of the arm's elevation, then glide on the rib cage about that pin
      const rv = matToRotvec(mul3(tr3(Rthup), Rarm));
      const Rsc = glide(ac, mul3(S.R, limitRotation(rodrigues(rv, Math.hypot(...rv) / 3), ac.limit_deg)), head, S, atlas.byName.get(ac.child).pos, gh.center);
      F[ac.child] = mkFrame(Rsc, gh.center, head, ks);
      // clavicle: from the sternoclavicular joint to where the scapula put the acromioclavicular joint
      const pJ = mapPoint(S, sc.center), acLive = mapPoint(F[ac.child], ac.center), tgt = [acLive[0] - pJ[0], acLive[1] - pJ[1], acLive[2] - pJ[2]];
      const rest = [ac.center[0] - sc.center[0], ac.center[1] - sc.center[1], ac.center[2] - sc.center[2]];
      const a = [0, 1, 2].map(i => S.R[i * 3] * rest[0] + S.R[i * 3 + 1] * rest[1] + S.R[i * 3 + 2] * rest[2]);
      const kcl = Math.min(1.35 * ks, Math.max(0.75 * ks, Math.hypot(...tgt) / Math.hypot(...rest)));
      F[sc.child] = mkFrame(mul3(aim(a, tgt, sc.limit_deg), S.R), sc.center, pJ, kcl);
      // humerus: on the model's skeleton
      F[gh.child] = mkFrame(mul3(Rsc, limitRotation(mul3(tr3(Rsc), Rarm), gh.limit_deg)), gh.center, head, T[SEG["uparm_" + k]].s || 1);
    } else if (S && sc && ac) {
      const pJ = mapPoint(S, sc.center);
      // clavicle: aim its lateral end (AC joint) at where the shoulder cap went, about the SC joint
      const cap = mapPoint(T[SEG["shoulder_" + k]], ac.center), tgt = [cap[0] - pJ[0], cap[1] - pJ[1], cap[2] - pJ[2]];
      const rest = [ac.center[0] - sc.center[0], ac.center[1] - sc.center[1], ac.center[2] - sc.center[2]];
      const a = [0, 1, 2].map(i => S.R[i * 3] * rest[0] + S.R[i * 3 + 1] * rest[1] + S.R[i * 3 + 2] * rest[2]);
      const Rcl = mul3(aim(a, tgt, sc.limit_deg), S.R);
      F[sc.child] = mkFrame(Rcl, sc.center, pJ, S.s || 1);
      // scapula: pinned at the AC joint; rotates with a third of the arm's elevation (scapulohumeral rhythm)
      const pJac = mapPoint(F[sc.child], ac.center);
      const rv = matToRotvec(mul3(tr3(Rthup), Rarm));
      const Rsc = glide(ac, mul3(S.R, limitRotation(rodrigues(rv, Math.hypot(...rv) / 3), ac.limit_deg)), pJac, S, atlas.byName.get(ac.child).pos);
      F[ac.child] = mkFrame(Rsc, ac.center, pJac, S.s || 1);
      if (gh) {                                                // humerus: ball pinned in the glenoid
        const Rh = mul3(Rsc, limitRotation(mul3(tr3(Rsc), Rarm), gh.limit_deg));
        F[gh.child] = mkFrame(Rh, gh.center, mapPoint(F[ac.child], gh.center), T[SEG["uparm_" + k]].s || 1);
      }
    }
    if (hip) {                                                 // femur: ball pinned in the acetabulum
      const Tp = T[SEG.pelvis];
      F[hip.child] = mkFrame(mul3(Tp.R, limitRotation(mul3(tr3(Tp.R), Rthigh), hip.limit_deg)), hip.center, mapPoint(Tp, hip.center), T[SEG["thigh_" + k]].s || 1);
    }
  }
  const ao = J.ao;
  if (ao && chain.includes(ao.parent)) {                       // skull on C1
    const c1 = cf[chain.indexOf(ao.parent)], Rh = T[SEG.head].R;
    F.skull = mkFrame(mul3(c1.R, limitRotation(mul3(tr3(c1.R), Rh), ao.limit_deg)), ao.center, mapPoint(headFrame || T[SEG.head], ao.center), (headFrame || T[SEG.head]).s || 1);   // the skull rides the cranium, not C1 (see runtime.py)
  }
  return F;
}

// Transform one structure's vertices into P (positions) and, if N/rn are given, normals.
// f = +1 keeps the raw MHR frame; f = -1 writes the display frame (x, -y, -z).
export function transformStructure(s, fr, P, N, rn, f = 1) {
  const p = s.pos, { T, cf, soft, F } = fr;
  if (s.bind === "soft") {
    for (let v = 0; v < p.length; v += 3) {
      let ox = 0, oy = 0, oz = 0, nx = 0, ny = 0, nz = 0;
      for (let k = 0; k < 3; k++) {
        const w = s.infW[v + k]; if (w <= 0) continue;
        const fr_ = soft[s.infIdx[v + k]], R = fr_.R, sc = fr_.s || 1;
        const x = sc * (p[v] - fr_.c[0]), y = sc * (p[v + 1] - fr_.c[1]), z = sc * (p[v + 2] - fr_.c[2]);
        ox += w * (R[0] * x + R[1] * y + R[2] * z + fr_.cp[0]);
        oy += w * (R[3] * x + R[4] * y + R[5] * z + fr_.cp[1]);
        oz += w * (R[6] * x + R[7] * y + R[8] * z + fr_.cp[2]);
        if (N) {
          nx += w * (R[0] * rn[v] + R[1] * rn[v + 1] + R[2] * rn[v + 2]);
          ny += w * (R[3] * rn[v] + R[4] * rn[v + 1] + R[5] * rn[v + 2]);
          nz += w * (R[6] * rn[v] + R[7] * rn[v + 1] + R[8] * rn[v + 2]);
        }
      }
      P[v] = ox; P[v + 1] = f * oy; P[v + 2] = f * oz;
      if (N) { N[v] = nx; N[v + 1] = f * ny; N[v + 2] = f * nz; }
    }
    return;
  }
  if (s.cartOwner && F.sternum) {                      // flexible costal cartilage (see prepareCartilage)
    const S = F.sternum;
    for (let v = 0; v < p.length; v += 3) {
      const Fr = F[s.cartOwner[v / 3]], w = s.cartW[v / 3], R = Fr.R, c = Fr.c, cp = Fr.cp, k1 = Fr.s || 1, k2 = S.s || 1;
      const x = k1 * (p[v] - c[0]), y = k1 * (p[v + 1] - c[1]), z = k1 * (p[v + 2] - c[2]);
      const X = R[0] * x + R[1] * y + R[2] * z + cp[0], Y = R[3] * x + R[4] * y + R[5] * z + cp[1], Z = R[6] * x + R[7] * y + R[8] * z + cp[2];
      const a = k2 * (p[v] - S.c[0]), b = k2 * (p[v + 1] - S.c[1]), d = k2 * (p[v + 2] - S.c[2]), Q = S.R;
      const X2 = Q[0] * a + Q[1] * b + Q[2] * d + S.cp[0], Y2 = Q[3] * a + Q[4] * b + Q[5] * d + S.cp[1], Z2 = Q[6] * a + Q[7] * b + Q[8] * d + S.cp[2];
      P[v] = (1 - w) * X + w * X2; P[v + 1] = f * ((1 - w) * Y + w * Y2); P[v + 2] = f * ((1 - w) * Z + w * Z2);
      if (N) {                                          // normals: blend of the two rotations
        const nx = (1 - w) * (R[0] * rn[v] + R[1] * rn[v + 1] + R[2] * rn[v + 2]) + w * (Q[0] * rn[v] + Q[1] * rn[v + 1] + Q[2] * rn[v + 2]);
        const ny = (1 - w) * (R[3] * rn[v] + R[4] * rn[v + 1] + R[5] * rn[v + 2]) + w * (Q[3] * rn[v] + Q[4] * rn[v + 1] + Q[5] * rn[v + 2]);
        const nz = (1 - w) * (R[6] * rn[v] + R[7] * rn[v + 1] + R[8] * rn[v + 2]) + w * (Q[6] * rn[v] + Q[7] * rn[v + 1] + Q[8] * rn[v + 2]);
        N[v] = nx; N[v + 1] = f * ny; N[v + 2] = f * nz;
      }
    }
    return;
  }
  let R, c, cp, k = 1;                                // k: uniform scale about the frame's centre
  let axis = null, kAxis = 1;                         // a different scale along one rest-frame axis (limb length vs thickness)
  let jaw = null;                                     // jaw hinge: rotation about the line through the condyles, before the frame
  if (s.jaw && fr.jawRad) {
    const ax = s.jaw.axis, M = rodrigues(ax, fr.jawRad);
    jaw = { M, p: s.jaw.pivot };
  }
  switch (s.bind) {
    case "mhr": {
      const fj = mhrFrame(fr, s.mhr);
      if (!fj) { P.set(p); if (N) N.set(rn); if (f !== 1) for (let v = 0; v < p.length; v += 3) { P[v + 1] *= f; P[v + 2] *= f; if (N) { N[v + 1] *= f; N[v + 2] *= f; } } return; }
      R = fj.R; c = fj.c; cp = fj.cp; k = fj.s; if (fj.ax) { axis = fj.ax; kAxis = fj.sl; } break;
    }
    case "joint": { const fr_ = F[s.name]; R = fr_.R; c = fr_.c; cp = fr_.cp; k = fr_.s || 1; break; }
    case "frame": { const fr_ = F[s.frame]; R = fr_.R; c = fr_.c; cp = fr_.cp; k = fr_.s || 1; break; }
    case "attach_ribs": { const t = rigidFromFrames(s.attach_ribs.filter(([r]) => F[r]).map(([r, w]) => [F[r], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; k = t.s; break; }
    case "vertebra": { const t = rigidFromFrames([[cf[s.chain_index], 1]], s.centroid); R = t.R; c = s.centroid; cp = t.cp; k = t.s; break; }
    case "attach": { const t = rigidFromFrames(s.attach.map(([i, w]) => [cf[i], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; k = t.s; break; }
    default: { const t = rigidFromFrames(s.bone_w.map(([sg, w]) => [T[sg], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; k = t.s; }
  }
  for (let v = 0; v < p.length; v += 3) {
    let px = p[v], py = p[v + 1], pz = p[v + 2];
    if (jaw) {
      const M = jaw.M, a = px - jaw.p[0], b = py - jaw.p[1], d = pz - jaw.p[2];
      px = jaw.p[0] + M[0] * a + M[1] * b + M[2] * d; py = jaw.p[1] + M[3] * a + M[4] * b + M[5] * d; pz = jaw.p[2] + M[6] * a + M[7] * b + M[8] * d;
    }
    let x = k * (px - c[0]), y = k * (py - c[1]), z = k * (pz - c[2]);
    if (axis) {
      const d = (kAxis - k) * ((px - c[0]) * axis[0] + (py - c[1]) * axis[1] + (pz - c[2]) * axis[2]);
      x += d * axis[0]; y += d * axis[1]; z += d * axis[2];
    }
    P[v] = R[0] * x + R[1] * y + R[2] * z + cp[0];
    P[v + 1] = f * (R[3] * x + R[4] * y + R[5] * z + cp[1]);
    P[v + 2] = f * (R[6] * x + R[7] * y + R[8] * z + cp[2]);
    if (N) {
      N[v] = R[0] * rn[v] + R[1] * rn[v + 1] + R[2] * rn[v + 2];
      N[v + 1] = f * (R[3] * rn[v] + R[4] * rn[v + 1] + R[5] * rn[v + 2]);
      N[v + 2] = f * (R[6] * rn[v] + R[7] * rn[v + 1] + R[8] * rn[v + 2]);
    }
  }
}

export class AnatomyLayer {
  // flipYZ: write display positions as (x, -y, -z) (mobile page). Leave false to keep the raw MHR
  // frame as a child of a mesh that already carries the flip (webcam page's bodyMesh).
  constructor(THREE, atlas, { flipYZ = false, opacity = 0.9, spine = "physio", supportStride = 1 } = {}) {
    this.spine = spine; this.supportStride = supportStride;
    this.THREE = THREE;
    this.atlas = atlas;
    this.flipYZ = flipYZ;
    this.group = new THREE.Group();
    this.group.name = "anatomy";
    this.byName = new Map();
    this.entries = atlas.structures.map(s => {
      const g = new THREE.BufferGeometry();
      g.setIndex(new THREE.BufferAttribute(s.faces, 1));
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(s.pos.length), 3));
      g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(s.pos.length), 3));
      const tmp = new THREE.BufferGeometry();       // rest normals once; rotated per frame
      tmp.setIndex(new THREE.BufferAttribute(s.faces, 1));
      tmp.setAttribute("position", new THREE.BufferAttribute(s.pos, 3));
      tmp.computeVertexNormals();
      const restN = new Float32Array(tmp.attributes.normal.array);
      tmp.dispose();
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(...s.color), transparent: opacity < 1, opacity, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(g, mat);
      mesh.frustumCulled = false; mesh.visible = false; mesh.name = s.name;
      this.group.add(mesh);
      const e = { s, mesh, restN };
      this.byName.set(s.name, e);
      return e;
    });
    this.group.visible = false;
    this.visibleGroups = new Set(["skull", "spine", "ribcage", "lung", "heart", "gi", "urinary"]);
    this._applyVisibility();
  }

  groups() {
    const m = new Map();
    for (const { s } of this.entries) m.set(s.group, (m.get(s.group) || 0) + 1);
    return [...m].map(([name, count]) => ({ name, count }));
  }
  setGroupVisible(name, on) { on ? this.visibleGroups.add(name) : this.visibleGroups.delete(name); this._applyVisibility(); }
  setOpacity(a) { for (const { mesh } of this.entries) { mesh.material.opacity = a; mesh.material.transparent = a < 1; mesh.material.needsUpdate = true; } }
  setEnabled(on) { this.group.visible = on; }
  _applyVisibility() { for (const { s, mesh } of this.entries) mesh.visible = this.visibleGroups.has(s.group) && !(this._undriven && s.bind === "mhr"); }

  // skin: Float32Array(V*3) live MHR vertices in the raw frame (metres), same layout as the mesh.
  // breath: 0..1 respiratory state (ribs, sternum and the structures they carry), depth: 1 = quiet breathing.
  // mhrJoints: optional { pos, rot } of the MHR skeleton (see computeFrames); used when this.useJoints is on.
  update(skin, breath = 0, depth = 1, mhrJoints = null) {
    if ((!this.group.visible && !this.xrayActive && !this.ctActive) || !skin) return;
    const fr = computeFrames(this.atlas, skin, breath, depth, this.spine, this.supportStride, this.useJoints === false ? null : mhrJoints, this.bodyScale !== false), f = this.flipYZ ? -1 : 1;
    fr.jawRad = (this.jawDeg || 0) * Math.PI / 180;        // jaw opening from the face tracker (degrees), 0 = closed
    this.lastFrames = fr;                                  // the frames of this pose (the CT volume view warps with them)
    const undriven = !fr.mhr;                              // bones bound to MHR joints cannot be placed without the joints
    if (undriven !== this._undriven) { this._undriven = undriven; this._applyVisibility(); }
    for (const e of this.entries) {
      if (undriven && e.s.bind === "mhr") continue;
      if (!(e.mesh.visible && this.group.visible) && !e.forceUpdate && !e.ctForce) continue;
      const g = e.mesh.geometry;
      transformStructure(e.s, fr, g.attributes.position.array, g.attributes.normal.array, e.restN, f);
      g.attributes.position.needsUpdate = true;
      g.attributes.normal.needsUpdate = true;
    }
  }
}
