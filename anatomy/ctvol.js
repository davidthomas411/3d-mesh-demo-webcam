// Live slice of the atlas subject's REAL CT, deformed by the rig.
//
// The CT has been resampled once into the rig's rest frame (tools/build_ct_volume.py -> atlas/ct_volume.*, half-float HU). The rig
// moves soft tissue with the two spine-chain frames that bracket its rest height (pelvis, the vertebrae, head), and the thighs with
// their own frames (anatomy/rig.py: soft_frame_weights). So for a pixel of the slice, which is a point p of the POSED body, the rest
// point y it came from satisfies  y = sum_k w_k(y) F_k^-1 p  with weights that depend only on y's height. The fragment shader solves
// that by fixed-point iteration (start from the pelvis frame, re-blend a few times) and samples the volume at y. Nothing is
// precomputed per pose: it follows the same frames the bones are drawn with, every frame.
//
// What it is not: the CT of the person in front of the camera. It is the atlas subject's CT, carried by the body model's motion
// and scaled per segment. Bones follow the soft-tissue frames here, so the shoulder girdle does not articulate in this view.
const MAX_FRAMES = 32;

export async function loadCtVolume(jsonUrl, binUrl) {
  const meta = await fetch(jsonUrl).then(r => { if (!r.ok) throw new Error("ct volume json " + r.status); return r.json(); });
  const buf = await fetch(binUrl).then(r => { if (!r.ok) throw new Error("ct volume bin " + r.status); return r.arrayBuffer(); });
  return { meta, data: new Uint16Array(buf) };          // IEEE half floats, x fastest
}

export class CtVolumeView {
  // layer: the AnatomyLayer of the CT atlas (its last computed frames drive the warp).
  constructor(THREE, renderer, layer, volume, { halfExtentM = 0.55, anchor = "vertebrae_T8" } = {}) {
    this.THREE = THREE; this.renderer = renderer; this.layer = layer; this.half = halfExtentM;
    const { meta, data } = volume, [nx, ny, nz] = meta.shape;
    if (data.length !== nx * ny * nz) throw new Error("ct volume size mismatch");
    this.meta = meta;
    const tex = new THREE.Data3DTexture(data, nx, ny, nz);
    tex.format = THREE.RedFormat; tex.type = THREE.HalfFloatType; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping; tex.unpackAlignment = 1; tex.needsUpdate = true;
    this.anchor = layer.byName.get(anchor) || null;
    if (this.anchor) this.anchor.ctForce = true;         // keep the anchor bone posed even when the bones are hidden
    const levels = new Float32Array(MAX_FRAMES); levels.set(meta.levels_y);
    this.nChain = meta.levels_y.length;                  // pelvis + vertebrae + head
    const mats = [], vec = () => Array.from({ length: MAX_FRAMES }, () => new THREE.Vector3());
    for (let i = 0; i < MAX_FRAMES; i++) mats.push(new THREE.Matrix3());
    this.uniforms = {
      uVol: { value: tex }, uOrigin: { value: new THREE.Vector3(...meta.origin) },
      uExtent: { value: new THREE.Vector3(nx * meta.voxel, ny * meta.voxel, nz * meta.voxel) },
      uR: { value: mats }, uC: { value: vec() }, uCp: { value: vec() }, uS: { value: new Float32Array(MAX_FRAMES).fill(1) },
      uLevels: { value: levels }, uNChain: { value: this.nChain }, uHipY: { value: meta.hip_y }, uHasFrames: { value: 0 },
      uCentre: { value: new THREE.Vector2() }, uHalf: { value: halfExtentM }, uZ: { value: 0 }, uWindow: { value: new THREE.Vector2(40, 400) },
      uPlaneFrame: { value: -1 },                          // >= 0: the slice plane is carried by this frame (body-aligned)
    };
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false, uniforms: this.uniforms,
      vertexShader: "out vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: `
        precision highp float; precision highp sampler3D;
        const int MAXF = ${MAX_FRAMES};
        uniform sampler3D uVol; uniform vec3 uOrigin; uniform vec3 uExtent;
        uniform mat3 uR[MAXF]; uniform vec3 uC[MAXF]; uniform vec3 uCp[MAXF]; uniform float uS[MAXF];
        uniform float uLevels[MAXF]; uniform int uNChain; uniform float uHipY; uniform float uHasFrames;
        uniform vec2 uCentre; uniform float uHalf; uniform float uZ; uniform vec2 uWindow; uniform int uPlaneFrame;
        in vec2 vUv; out vec4 outColor;
        // rest point that frame k sends to p:  p = s R (y - c) + cp
        vec3 back(int k, vec3 p){ return uC[k] + (transpose(uR[k]) * (p - uCp[k])) / uS[k]; }
        vec3 toRest(vec3 p){
          vec3 y = back(0, p);
          for (int it = 0; it < 10; it++) {                // the neck, where neighbouring frames differ most, needs the most
            int lo = 0;                                   // chain frames bracketing the rest height (y is down: higher = smaller y)
            for (int i = 1; i < uNChain - 1; i++) if (y.y < uLevels[i]) lo = i;
            float t = clamp((uLevels[lo] - y.y) / max(uLevels[lo] - uLevels[lo + 1], 1e-5), 0.0, 1.0);
            vec3 yn = mix(back(lo, p), back(lo + 1, p), t);
            // thighs: below the hip joint the leg's own frame takes over, left or right
            float wt = smoothstep(0.0, 1.0, (y.y - (uHipY - 0.03)) / 0.12);
            if (wt > 0.0) yn = mix(yn, back(uNChain + (y.x > 0.0 ? 2 : 3), p), wt);
            y = yn;
          }
          return y;
        }
        void main(){
          // same picture as the label slice and the DRR: seen from the front, world +x (the patient's left) to the image right, head up
          vec3 p = vec3(uCentre.x + (vUv.x - 0.5) * 2.0 * uHalf, uCentre.y - (vUv.y - 0.5) * 2.0 * uHalf, uZ);
          // body-aligned: the plane is a coronal plane of the REST body carried by one trunk frame, so it stays coronal when the
          // person turns, while bending, twisting and head motion still deform what it cuts
          if (uPlaneFrame >= 0 && uHasFrames > 0.5) p = uCp[uPlaneFrame] + uS[uPlaneFrame] * (uR[uPlaneFrame] * (p - uC[uPlaneFrame]));
          vec3 y = uHasFrames > 0.5 ? toRest(p) : p;
          vec3 uvw = (y - uOrigin) / uExtent;
          float hu = -1000.0;
          if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) hu = texture(uVol, uvw).r;
          float g = clamp((hu - (uWindow.x - 0.5 * uWindow.y)) / uWindow.y, 0.0, 1.0);
          outColor = vec4(vec3(g), 1.0);
        }` });
    this.scene = new THREE.Scene(); this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));
  }

  setWindow(level, width) { this.uniforms.uWindow.value.set(level, width); }

  // Front-to-back position (raw frame z, front is -z) of the anchor bone in its current pose.
  anchorZ() {
    if (!this.anchor) return 0;
    const a = this.anchor.mesh.geometry.attributes.position.array;
    let z = 0;
    for (let i = 2; i < a.length; i += 3) z += a[i];
    return z / (a.length / 3);
  }

  // centre: [x, y] picture centre (raw MHR frame, metres); offsetM: slice position relative to the anchor bone (negative = anterior).
  // bodyAligned: cut a coronal plane of the rest body carried by the anchor vertebra's frame (centre is then ignored: the picture
  // is centred on the rest body); otherwise the plane is fixed to the camera, like the DRR.
  render(centre = [0, -1.3], offsetM = -0.05, viewport = null, bodyAligned = true) {
    const { renderer, uniforms: u } = this, fr = this.layer.lastFrames;
    if (fr && fr.soft) {
      const n = Math.min(fr.soft.length, MAX_FRAMES);
      for (let k = 0; k < n; k++) {
        const f = fr.soft[k], R = f.R;
        u.uR.value[k].set(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);     // row-major in, so the shader's uR * v is R v
        u.uC.value[k].set(f.c[0], f.c[1], f.c[2]); u.uCp.value[k].set(f.cp[0], f.cp[1], f.cp[2]); u.uS.value[k] = f.s || 1;
      }
      u.uHasFrames.value = 1;
    }
    const k = this.anchor ? 1 + this.layer.atlas.chain.indexOf(this.anchor.s.name) : 0;      // soft frames: pelvis, then the chain
    if (bodyAligned && k > 0) {
      u.uPlaneFrame.value = k; u.uCentre.value.set(0, -1.3); u.uZ.value = this.anchor.s.centroid[2] + offsetM;
    } else {
      u.uPlaneFrame.value = -1; u.uCentre.value.set(centre[0], centre[1]); u.uZ.value = this.anchorZ() + offsetM;
    }
    u.uHalf.value = this.half;
    if (viewport) { renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissorTest(true); }
    const autoClear = renderer.autoClear; renderer.autoClear = false;
    renderer.render(this.scene, this.cam);
    renderer.autoClear = autoClear;
    if (viewport) renderer.setScissorTest(false);
  }
}
