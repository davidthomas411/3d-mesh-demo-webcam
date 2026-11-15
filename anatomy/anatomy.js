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
  return { meta, segments, structures, chain: meta.chain || [] };
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

  // Segment frames, spine-chain frames and the soft-tissue frame list for one live skin mesh.
  _frames(skin) {
    const A = this.atlas, SEG = Object.fromEntries(A.segments.map((g, i) => [g.name, i]));
    const T = A.segments.map(g => {
      const r = weightedRigid(g.rest, g.idx, null, skin);
      return { q: r.q, R: r.R, c: r.cs, cp: r.cd };
    });
    const chain = A.chain, n = chain.length, cf = [];
    if (n) {
      const q = [], c = [], sp = [];
      for (const name of chain) {
        const s = this.byName.get(name).s, wf = s.bone_w.map(([sg, w]) => [T[sg], w]);
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
    const soft = [T[SEG.pelvis], ...cf, T[SEG.head], T[SEG.uparm_l], T[SEG.uparm_r], T[SEG.thigh_l], T[SEG.thigh_r]];
    return { T, cf, soft };
  }

  // skin: Float32Array(V*3) live MHR vertices in the raw frame (metres), same layout as the mesh.
  update(skin) {
    if (!this.group.visible || !skin) return;
    const { T, cf, soft } = this._frames(skin);
    const f = this.flipYZ ? -1 : 1;
    for (const e of this.entries) {
      if (!e.mesh.visible) continue;
      const { s } = e, P = e.mesh.geometry.attributes.position.array, N = e.mesh.geometry.attributes.normal.array;
      const p = s.pos, rn = e.restN;
      if (s.bind === "soft") {
        for (let v = 0; v < p.length; v += 3) {
          let ox = 0, oy = 0, oz = 0, nx = 0, ny = 0, nz = 0;
          for (let k = 0; k < 3; k++) {
            const w = s.infW[v + k]; if (w <= 0) continue;
            const fr = soft[s.infIdx[v + k]], R = fr.R;
            const x = p[v] - fr.c[0], y = p[v + 1] - fr.c[1], z = p[v + 2] - fr.c[2];
            ox += w * (R[0] * x + R[1] * y + R[2] * z + fr.cp[0]);
            oy += w * (R[3] * x + R[4] * y + R[5] * z + fr.cp[1]);
            oz += w * (R[6] * x + R[7] * y + R[8] * z + fr.cp[2]);
            nx += w * (R[0] * rn[v] + R[1] * rn[v + 1] + R[2] * rn[v + 2]);
            ny += w * (R[3] * rn[v] + R[4] * rn[v + 1] + R[5] * rn[v + 2]);
            nz += w * (R[6] * rn[v] + R[7] * rn[v + 1] + R[8] * rn[v + 2]);
          }
          P[v] = ox; P[v + 1] = f * oy; P[v + 2] = f * oz; N[v] = nx; N[v + 1] = f * ny; N[v + 2] = f * nz;
        }
      } else {
        let wf;
        if (s.bind === "vertebra") wf = [[cf[s.chain_index], 1]];
        else if (s.bind === "attach") wf = s.attach.map(([i, w]) => [cf[i], w]);
        else wf = s.bone_w.map(([sg, w]) => [T[sg], w]);
        const { R, cp } = rigidFromFrames(wf, s.centroid), c = s.centroid;
        for (let v = 0; v < p.length; v += 3) {
          const x = p[v] - c[0], y = p[v + 1] - c[1], z = p[v + 2] - c[2];
          P[v] = R[0] * x + R[1] * y + R[2] * z + cp[0];
          P[v + 1] = f * (R[3] * x + R[4] * y + R[5] * z + cp[1]);
          P[v + 2] = f * (R[6] * x + R[7] * y + R[8] * z + cp[2]);
          N[v] = R[0] * rn[v] + R[1] * rn[v + 1] + R[2] * rn[v + 2];
          N[v + 1] = f * (R[3] * rn[v] + R[4] * rn[v + 1] + R[5] * rn[v + 2]);
          N[v + 2] = f * (R[6] * rn[v] + R[7] * rn[v + 1] + R[8] * rn[v + 2]);
        }
      }
      e.mesh.geometry.attributes.position.needsUpdate = true;
      e.mesh.geometry.attributes.normal.needsUpdate = true;
    }
  }
}
