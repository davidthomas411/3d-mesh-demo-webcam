// Anatomy layer: TotalSegmentator-class structures driven by a live MHR skin mesh.
// Port of anatomy/runtime.py. ES module, no dependencies besides three.js (passed in).
//
// Each structure moves by the weighted rigid transform that carries its supporting skin
// vertices from the MHR rest pose to the live mesh. This is ILLUSTRATIVE atlas anatomy, not
// patient-specific, and must be presented as such (see DISCLAIMER).

export const DISCLAIMER = "Atlas-based illustrative anatomy. Not patient-specific. Not for clinical use.";

export async function loadAtlas(jsonUrl, binUrl) {
  const [meta, buf] = await Promise.all([
    fetch(jsonUrl).then(r => { if (!r.ok) throw new Error("atlas json " + r.status); return r.json(); }),
    fetch(binUrl).then(r => { if (!r.ok) throw new Error("atlas bin " + r.status); return r.arrayBuffer(); }),
  ]);
  const structures = meta.structures.map(e => {
    const n = e.n_verts, ns = e.support_n;
    const idx = e.idx_type === "u16"
      ? Uint32Array.from(new Uint16Array(buf, e.idx, e.n_tris * 3))
      : new Uint32Array(buf.slice(e.idx, e.idx + e.n_tris * 12));
    return {
      ...e,
      pos: new Float32Array(buf.slice(e.pos, e.pos + n * 12)),
      faces: idx,
      sIdx: Uint32Array.from(new Uint16Array(buf, e.support_idx, ns)),
      sW: new Float32Array(buf.slice(e.support_w, e.support_w + ns * 4)),
      sRest: new Float32Array(buf.slice(e.support_rest, e.support_rest + ns * 12)),
    };
  });
  return { meta, structures };
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
  const n = sIdx.length;
  let wsum = 0;
  for (let i = 0; i < n; i++) wsum += sW[i];
  const cs = [0, 0, 0], cd = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = sW[i] / wsum, j = sIdx[i] * 3;
    for (let k = 0; k < 3; k++) { cs[k] += w * sRest[i * 3 + k]; cd[k] += w * live[j + k]; }
  }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // S[a][b] = sum w (src_a - cs_a)(dst_b - cd_b)
  for (let i = 0; i < n; i++) {
    const w = sW[i] / wsum, j = sIdx[i] * 3;
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
  return { R, cs, cd };
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
    this.group.frustumCulled = false;
    this.entries = atlas.structures.map(s => {
      const g = new THREE.BufferGeometry();
      g.setIndex(new THREE.BufferAttribute(s.faces, 1));
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(s.pos.length), 3));
      g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(s.pos.length), 3));
      // rest normals once (rigid motion keeps them valid: just rotate per frame)
      const tmp = new THREE.BufferGeometry();
      tmp.setIndex(new THREE.BufferAttribute(s.faces, 1));
      tmp.setAttribute("position", new THREE.BufferAttribute(s.pos, 3));
      tmp.computeVertexNormals();
      const restN = new Float32Array(tmp.attributes.normal.array);
      tmp.dispose();
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(...s.color), transparent: opacity < 1, opacity,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(g, mat);
      mesh.frustumCulled = false;
      mesh.visible = false;
      mesh.name = s.name;
      mesh.userData.structure = s.name;
      this.group.add(mesh);
      return { s, mesh, restN, R: new Float64Array(9) };
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

  setGroupVisible(name, on) {
    on ? this.visibleGroups.add(name) : this.visibleGroups.delete(name);
    this._applyVisibility();
  }

  setOpacity(a) {
    for (const { mesh } of this.entries) { mesh.material.opacity = a; mesh.material.transparent = a < 1; mesh.material.needsUpdate = true; }
  }

  setEnabled(on) { this.group.visible = on; }

  _applyVisibility() {
    for (const { s, mesh } of this.entries) mesh.visible = this.visibleGroups.has(s.group);
  }

  // skin: Float32Array(V*3) live MHR vertices in the raw frame (metres), same layout as the mesh.
  update(skin) {
    if (!this.group.visible || !skin) return;
    const f = this.flipYZ ? -1 : 1;
    for (const e of this.entries) {
      if (!e.mesh.visible) continue;
      const { s } = e;
      const { R, cs, cd } = weightedRigid(s.sRest, s.sIdx, s.sW, skin, e.R);
      const P = e.mesh.geometry.attributes.position.array, N = e.mesh.geometry.attributes.normal.array;
      const p = s.pos, rn = e.restN;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i] - cs[0], y = p[i + 1] - cs[1], z = p[i + 2] - cs[2];
        P[i] = R[0] * x + R[1] * y + R[2] * z + cd[0];
        P[i + 1] = f * (R[3] * x + R[4] * y + R[5] * z + cd[1]);
        P[i + 2] = f * (R[6] * x + R[7] * y + R[8] * z + cd[2]);
        const nx = rn[i], ny = rn[i + 1], nz = rn[i + 2];
        N[i] = R[0] * nx + R[1] * ny + R[2] * nz;
        N[i + 1] = f * (R[3] * nx + R[4] * ny + R[5] * nz);
        N[i + 2] = f * (R[6] * nx + R[7] * ny + R[8] * nz);
      }
      e.mesh.geometry.attributes.position.needsUpdate = true;
      e.mesh.geometry.attributes.normal.needsUpdate = true;
    }
  }
}
