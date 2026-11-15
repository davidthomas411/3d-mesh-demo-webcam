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
  return { meta, segments, structures, byName, chain: meta.chain || [], joints: meta.joints || [] };
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
const mapPoint = (f, p) => {
  const x = p[0] - f.c[0], y = p[1] - f.c[1], z = p[2] - f.c[2], R = f.R;
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
  if (frames.length === 1) { const f = frames[0][0]; return { R: f.R, cp: mapPoint(f, centroid) }; }
  const R = quatToR(blendQuat(frames)), cp = [0, 0, 0];
  for (const [f, w] of frames) { const m = mapPoint(f, centroid); for (let k = 0; k < 3; k++) cp[k] += w * m[k]; }
  return { R, cp };
}
function blendFrame(centroid, wf) {
  const q = blendQuat(wf), cp = [0, 0, 0];
  for (const [f, w] of wf) { const m = mapPoint(f, centroid); for (let k = 0; k < 3; k++) cp[k] += w * m[k]; }
  return { q, R: quatToR(q), c: centroid, cp };
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
const mkFrame = (R, J, pJ) => ({ q: matToQuat(R), R, c: J, cp: pJ });

// ---------------------------------------------------------------------------------------------
// Kinematics (no three.js): segment frames -> spine chain -> joint frames -> soft-tissue frames.
// breath: 0 (end-expiration) .. 1 (end-inspiration); depth: 1 = quiet breathing.
export function computeFrames(atlas, skin, breath = 0, depth = 1) {
  const SEG = Object.fromEntries(atlas.segments.map((g, i) => [g.name, i]));
  const T = atlas.segments.map(g => {
    const r = weightedRigid(g.rest, g.idx, null, skin);
    return { q: r.q, R: r.R, c: r.cs, cp: r.cd };
  });
  const chain = atlas.chain, n = chain.length, cf = [];
  if (n) {
    const q = [], c = [], sp = [];
    for (const name of chain) {
      const s = atlas.byName.get(name), wf = s.bone_w.map(([sg, w]) => [T[sg], w]);
      q.push(blendQuat(wf)); c.push(s.centroid);
      const p = [0, 0, 0];
      for (const [f, w] of wf) { const m = mapPoint(f, s.centroid); for (let k = 0; k < 3; k++) p[k] += w * m[k]; }
      sp.push(p);
    }
    const cp = [sp[0].slice()];
    for (let i = 1; i < n; i++) {
      const d = q[i][0] * q[i - 1][0] + q[i][1] * q[i - 1][1] + q[i][2] * q[i - 1][2] + q[i][3] * q[i - 1][3];
      const sg = d >= 0 ? 1 : -1, qm = q[i - 1].map((v, k) => v + sg * q[i][k]), nn = Math.hypot(...qm);
      const R = quatToR(qm.map(v => v / nn)), dv = [c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1], c[i][2] - c[i - 1][2]];
      cp.push([0, 1, 2].map(k => cp[i - 1][k] + R[k * 3] * dv[0] + R[k * 3 + 1] * dv[1] + R[k * 3 + 2] * dv[2]));
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
      cf.push({ q: q[i], R: quatToR(q[i]), c: c[i], cp: [cp[i][0] + out[0], cp[i][1] + out[1], cp[i][2] + out[2]] });
    }
  }
  const F = jointFrames(atlas, T, cf, SEG, breath, depth);
  const soft = [T[SEG.pelvis], ...cf, T[SEG.head], T[SEG.uparm_l], T[SEG.uparm_r], T[SEG.thigh_l], T[SEG.thigh_r]];
  return { T, cf, soft, F };
}

// Body frames {q,R,c,cp} (x' = R (x - c) + cp) for every joint-driven bone. Mirrors _joint_frames in runtime.py.
function jointFrames(atlas, T, cf, SEG, breath, depth) {
  const chain = atlas.chain, J = Object.fromEntries(atlas.joints.map(j => [j.name, j])), F = {};
  for (const j of atlas.joints) {                              // ribs: costovertebral joint + respiratory rotation
    if (j.type !== "costovertebral" || !chain.includes(j.parent)) continue;
    const fv = cf[chain.indexOf(j.parent)], ang = j.resp_deg * depth * breath * DEG * j.resp_sign;
    F[j.child] = mkFrame(mul3(fv.R, rodrigues(j.axis, ang)), j.center, mapPoint(fv, j.center));
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
    if (S && sc && ac) {
      const pJ = mapPoint(S, sc.center);
      // clavicle: aim its lateral end (AC joint) at where the shoulder cap went, about the SC joint
      const cap = mapPoint(T[SEG["shoulder_" + k]], ac.center), tgt = [cap[0] - pJ[0], cap[1] - pJ[1], cap[2] - pJ[2]];
      const rest = [ac.center[0] - sc.center[0], ac.center[1] - sc.center[1], ac.center[2] - sc.center[2]];
      const a = [0, 1, 2].map(i => S.R[i * 3] * rest[0] + S.R[i * 3 + 1] * rest[1] + S.R[i * 3 + 2] * rest[2]);
      const Rcl = mul3(aim(a, tgt, sc.limit_deg), S.R);
      F[sc.child] = mkFrame(Rcl, sc.center, pJ);
      // scapula: pinned at the AC joint; rotates with a third of the arm's elevation (scapulohumeral rhythm)
      const pJac = mapPoint(F[sc.child], ac.center);
      const rv = matToRotvec(mul3(tr3(Rthup), Rarm));
      const Rsc = mul3(S.R, limitRotation(rodrigues(rv, Math.hypot(...rv) / 3), ac.limit_deg));
      F[ac.child] = mkFrame(Rsc, ac.center, pJac);
      if (gh) {                                                // humerus: ball pinned in the glenoid
        const Rh = mul3(Rsc, limitRotation(mul3(tr3(Rsc), Rarm), gh.limit_deg));
        F[gh.child] = mkFrame(Rh, gh.center, mapPoint(F[ac.child], gh.center));
      }
    }
    if (hip) {                                                 // femur: ball pinned in the acetabulum
      const Tp = T[SEG.pelvis];
      F[hip.child] = mkFrame(mul3(Tp.R, limitRotation(mul3(tr3(Tp.R), Rthigh), hip.limit_deg)), hip.center, mapPoint(Tp, hip.center));
    }
  }
  const ao = J.ao;
  if (ao && chain.includes(ao.parent)) {                       // skull on C1
    const c1 = cf[chain.indexOf(ao.parent)], Rh = T[SEG.head].R;
    F.skull = mkFrame(mul3(c1.R, limitRotation(mul3(tr3(c1.R), Rh), ao.limit_deg)), ao.center, mapPoint(c1, ao.center));
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
        const fr_ = soft[s.infIdx[v + k]], R = fr_.R;
        const x = p[v] - fr_.c[0], y = p[v + 1] - fr_.c[1], z = p[v + 2] - fr_.c[2];
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
  let R, c, cp;
  switch (s.bind) {
    case "joint": { const fr_ = F[s.name]; R = fr_.R; c = fr_.c; cp = fr_.cp; break; }
    case "frame": { const fr_ = F[s.frame]; R = fr_.R; c = fr_.c; cp = fr_.cp; break; }
    case "attach_ribs": { const t = rigidFromFrames(s.attach_ribs.filter(([r]) => F[r]).map(([r, w]) => [F[r], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; break; }
    case "vertebra": { const t = rigidFromFrames([[cf[s.chain_index], 1]], s.centroid); R = t.R; c = s.centroid; cp = t.cp; break; }
    case "attach": { const t = rigidFromFrames(s.attach.map(([i, w]) => [cf[i], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; break; }
    default: { const t = rigidFromFrames(s.bone_w.map(([sg, w]) => [T[sg], w]), s.centroid); R = t.R; c = s.centroid; cp = t.cp; }
  }
  for (let v = 0; v < p.length; v += 3) {
    const x = p[v] - c[0], y = p[v + 1] - c[1], z = p[v + 2] - c[2];
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
  constructor(THREE, atlas, { flipYZ = false, opacity = 0.9 } = {}) {
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
  _applyVisibility() { for (const { s, mesh } of this.entries) mesh.visible = this.visibleGroups.has(s.group); }

  // skin: Float32Array(V*3) live MHR vertices in the raw frame (metres), same layout as the mesh.
  // breath: 0..1 respiratory state (ribs, sternum and the structures they carry), depth: 1 = quiet breathing.
  update(skin, breath = 0, depth = 1) {
    if (!this.group.visible || !skin) return;
    const fr = computeFrames(this.atlas, skin, breath, depth), f = this.flipYZ ? -1 : 1;
    for (const e of this.entries) {
      if (!e.mesh.visible) continue;
      const g = e.mesh.geometry;
      transformStructure(e.s, fr, g.attributes.position.array, g.attributes.normal.array, e.restN, f);
      g.attributes.position.needsUpdate = true;
      g.attributes.normal.needsUpdate = true;
    }
  }
}
