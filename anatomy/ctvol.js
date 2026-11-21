// Live pictures of the atlas subject's REAL CT, deformed by the rig: a slice, and a DRR ray-marched through the voxels.
//
// The CT has been resampled once into the rig's rest frame (tools/build_ct_volume.py -> atlas/ct_volume.*, half-float HU). The rig
// moves soft tissue with the two spine-chain frames that bracket its rest height (pelvis, the vertebrae, head), and the thighs with
// their own frames (anatomy/rig.py: soft_frame_weights). So for a point p of the POSED body, the rest point y it came from
// satisfies  y = sum_k w_k(y) F_k^-1 p  with weights that depend only on y's height. The fragment shader solves that by
// fixed-point iteration (start from the pelvis frame, re-blend a few times) and samples the volume at y. Nothing is precomputed
// per pose: it follows the same frames the bones are drawn with, every frame.
//
// Slice: one sample per pixel. DRR: an orthographic ray per pixel (same camera as the surface DRR in xray.js, so it can be
// turned), marched front to back; each step re-uses the previous step's rest point as its starting guess, so two iterations are
// enough. Both are drawn into a small target first, which is what lets a captured reference be kept and shown in green under the
// live picture in magenta.
//
// PET (optional second volume, same rest frame): fused in colour on the slice, and as a maximum-intensity projection on the DRR.
//
// What it is not: the CT of the person in front of the camera. It is the atlas subject's CT, carried by the body model's motion
// and scaled per segment. Bones follow the soft-tissue frames here, so the shoulder girdle does not articulate in these views.
const MAX_FRAMES = 32;
const STEPS = 96;

export async function loadCtVolume(jsonUrl, binUrl) {
  const meta = await fetch(jsonUrl).then(r => { if (!r.ok) throw new Error("ct volume json " + r.status); return r.json(); });
  const buf = await fetch(binUrl).then(r => { if (!r.ok) throw new Error("ct volume bin " + r.status); return r.arrayBuffer(); });
  return { meta, data: new Uint16Array(buf) };          // IEEE half floats, x fastest
}

// PET in the same rest frame: uint8, SUV * meta.suv_scale, x fastest. Resolves to null when the files are not there.
export async function loadPetVolume(jsonUrl, binUrl) {
  try {
    const meta = await fetch(jsonUrl).then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); });
    const buf = await fetch(binUrl).then(r => { if (!r.ok) throw new Error(String(r.status)); return r.arrayBuffer(); });
    return { meta, data: new Uint8Array(buf) };
  } catch { return null; }
}

export class CtVolumeView {
  // layer: the AnatomyLayer of the CT atlas (its last computed frames drive the warp). pet: loadPetVolume() result or null.
  constructor(THREE, renderer, layer, volume, { halfExtentM = 0.55, anchor = "vertebrae_T8", pet = null, sliceSize = 512, drrSize = 320 } = {}) {
    this.THREE = THREE; this.renderer = renderer; this.layer = layer; this.half = halfExtentM;
    const { meta, data } = volume, [nx, ny, nz] = meta.shape;
    if (data.length !== nx * ny * nz) throw new Error("ct volume size mismatch");
    this.meta = meta;
    const tex = new THREE.Data3DTexture(data, nx, ny, nz);
    tex.format = THREE.RedFormat; tex.type = THREE.HalfFloatType; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping; tex.unpackAlignment = 1; tex.needsUpdate = true;
    let petTex;
    this.hasPet = !!pet; this.petOn = false; this.petMeta = pet ? pet.meta : null;
    if (pet) {
      const [px, py, pz] = pet.meta.shape;
      if (pet.data.length !== px * py * pz) throw new Error("pet volume size mismatch");
      petTex = new THREE.Data3DTexture(pet.data, px, py, pz);
    } else petTex = new THREE.Data3DTexture(new Uint8Array(1), 1, 1, 1);
    petTex.format = THREE.RedFormat; petTex.type = THREE.UnsignedByteType; petTex.minFilter = THREE.LinearFilter; petTex.magFilter = THREE.LinearFilter;
    petTex.wrapS = petTex.wrapT = petTex.wrapR = THREE.ClampToEdgeWrapping; petTex.unpackAlignment = 1; petTex.needsUpdate = true;
    this.anchor = layer.byName.get(anchor) || null;
    if (this.anchor) this.anchor.ctForce = true;         // keep the anchor bone posed even when the bones are hidden
    const levels = new Float32Array(MAX_FRAMES); levels.set(meta.levels_y);
    this.nChain = meta.levels_y.length;                  // pelvis + vertebrae + head
    const mats = [], vec = () => Array.from({ length: MAX_FRAMES }, () => new THREE.Vector3());
    for (let i = 0; i < MAX_FRAMES; i++) mats.push(new THREE.Matrix3());
    const petOrigin = pet ? pet.meta.origin : meta.origin;
    const petExtent = pet ? pet.meta.shape.map(n => n * pet.meta.voxel) : [1, 1, 1];
    this.uniforms = {
      uVol: { value: tex }, uOrigin: { value: new THREE.Vector3(...meta.origin) },
      uExtent: { value: new THREE.Vector3(nx * meta.voxel, ny * meta.voxel, nz * meta.voxel) },
      uPet: { value: petTex }, uPetOrigin: { value: new THREE.Vector3(...petOrigin) }, uPetExtent: { value: new THREE.Vector3(...petExtent) },
      uPetOn: { value: 0 }, uPetGain: { value: pet ? 255 / (pet.meta.suv_scale * (pet.meta.suv_window || 7)) : 1 },   // texture value -> fraction of the SUV window
      uR: { value: mats }, uC: { value: vec() }, uCp: { value: vec() }, uS: { value: new Float32Array(MAX_FRAMES).fill(1) },
      uLevels: { value: levels }, uNChain: { value: this.nChain }, uHipY: { value: meta.hip_y }, uCutY: { value: meta.cut_y ?? meta.hip_y + 0.04 }, uLegs: { value: meta.legs ? 1 : 0 }, uHasFrames: { value: 0 },
      uCentre: { value: new THREE.Vector3() }, uHalf: { value: halfExtentM }, uZ: { value: 0 }, uWindow: { value: new THREE.Vector2(40, 400) },
      uPlaneFrame: { value: -1 },                          // >= 0: the slice plane is carried by this frame (body-aligned)
      uRight: { value: new THREE.Vector3(1, 0, 0) }, uUp: { value: new THREE.Vector3(0, -1, 0) }, uFwd: { value: new THREE.Vector3(0, 0, 1) },
      uDepth: { value: 0.55 },                             // the ray is marched this far either side of the centre
    };
    const common = `
        precision highp float; precision highp sampler3D;
        const int MAXF = ${MAX_FRAMES};
        uniform sampler3D uVol; uniform vec3 uOrigin; uniform vec3 uExtent;
        uniform sampler3D uPet; uniform vec3 uPetOrigin; uniform vec3 uPetExtent; uniform float uPetOn; uniform float uPetGain;
        uniform mat3 uR[MAXF]; uniform vec3 uC[MAXF]; uniform vec3 uCp[MAXF]; uniform float uS[MAXF];
        uniform float uLevels[MAXF]; uniform int uNChain; uniform float uHipY; uniform float uCutY; uniform float uLegs; uniform float uHasFrames;
        uniform vec3 uCentre; uniform float uHalf; uniform float uZ; uniform vec2 uWindow; uniform int uPlaneFrame;
        uniform vec3 uRight; uniform vec3 uUp; uniform vec3 uFwd; uniform float uDepth;
        in vec2 vUv; out vec4 outColor;
        // rest point that frame k sends to p:  p = s R (y - c) + cp
        vec3 back(int k, vec3 p){ return uC[k] + (transpose(uR[k]) * (p - uCp[k])) / uS[k]; }
        // one re-blend: the frames are chosen by the current estimate y of the rest point
        vec3 refine(vec3 y, vec3 p){
          int lo = 0;                                     // chain frames bracketing the rest height (y is down: higher = smaller y)
          for (int i = 1; i < uNChain - 1; i++) if (y.y < uLevels[i]) lo = i;
          float t = clamp((uLevels[lo] - y.y) / max(uLevels[lo] - uLevels[lo + 1], 1e-5), 0.0, 1.0);
          vec3 yn = mix(back(lo, p), back(lo + 1, p), t);
          // thighs: below the hip joint the leg's own frame takes over, left or right. Only for a scan that includes the legs: one
          // that ends at the hips stays with the pelvis down to its end (a raised thigh otherwise smears the scan's end face)
          float wt = uLegs * smoothstep(0.0, 1.0, (y.y - (uHipY - 0.03)) / 0.12);
          if (wt > 0.0) yn = mix(yn, back(uNChain + (y.x > 0.0 ? 2 : 3), p), wt);
          return yn;
        }
        vec3 toRest(vec3 p){
          vec3 y = back(0, p);
          for (int it = 0; it < 10; it++) y = refine(y, p);   // the neck, where neighbouring frames differ most, needs the most
          return y;
        }
        float huAt(vec3 y){
          vec3 uvw = (y - uOrigin) / uExtent;
          // below the end of the scan there is nothing to show: without the cut, the scan's end face is smeared down the thighs
          bool inside = all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0))) && y.y < uCutY;
          float hu = inside ? texture(uVol, uvw).r : -1000.0;
          return hu;
        }
        float petAt(vec3 y){                               // fraction of the SUV window
          vec3 uvw = (y - uPetOrigin) / uPetExtent;
          bool inside = all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0))) && y.y < uCutY;
          float v = inside ? texture(uPet, uvw).r * uPetGain : 0.0;
          return v;
        }`;
    const mkMat = (main) => new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false, uniforms: this.uniforms,
      vertexShader: "out vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }", fragmentShader: common + main });
    // Both passes write (grey, pet, 0, 1) with grey and pet in 0..1.
    this.sliceMat = mkMat(`
        void main(){
          // same picture as the label slice and the DRR: seen from the front, world +x (the patient's left) to the image right, head up
          vec3 p = vec3(uCentre.x + (vUv.x - 0.5) * 2.0 * uHalf, uCentre.y - (vUv.y - 0.5) * 2.0 * uHalf, uZ);
          // body-aligned: the plane is a coronal plane of the REST body carried by one trunk frame, so it stays coronal when the
          // person turns, while bending, twisting and head motion still deform what it cuts
          if (uPlaneFrame >= 0 && uHasFrames > 0.5) p = uCp[uPlaneFrame] + uS[uPlaneFrame] * (uR[uPlaneFrame] * (p - uC[uPlaneFrame]));
          vec3 y = uHasFrames > 0.5 ? toRest(p) : p;
          float g = clamp((huAt(y) - (uWindow.x - 0.5 * uWindow.y)) / uWindow.y, 0.0, 1.0);
          outColor = vec4(g, uPetOn > 0.5 ? clamp(petAt(y), 0.0, 1.0) : 0.0, 0.0, 1.0);
        }`);
    this.drrMat = mkMat(`
        const int STEPS = ${STEPS};
        float hash(vec2 q){ return fract(sin(dot(q, vec2(12.9898, 78.233))) * 43758.5453); }
        void main(){
          vec3 o = uCentre + uRight * ((vUv.x - 0.5) * 2.0 * uHalf) + uUp * ((vUv.y - 0.5) * 2.0 * uHalf);
          float dt = 2.0 * uDepth / float(STEPS);
          vec3 p = o + uFwd * (-uDepth + dt * hash(gl_FragCoord.xy));       // jittered start: banding becomes fine noise, blurred away
          vec3 stepv = uFwd * dt;
          vec3 y = uHasFrames > 0.5 ? toRest(p) : p;
          float soft = 0.0, bone = 0.0, pet = 0.0;
          for (int i = 0; i < STEPS; i++) {
            float hu = huAt(y);
            if (hu > -900.0) {
              soft += clamp((hu + 1000.0) / 1000.0, 0.0, 1.1);              // relative electron density, about 1 for soft tissue
              bone += max(hu - 120.0, 0.0) / 1000.0;                        // the extra attenuation of bone at diagnostic energies
              if (uPetOn > 0.5) pet = max(pet, petAt(y));
            }
            p += stepv;
            if (uHasFrames > 0.5) { y = refine(y + stepv, p); y = refine(y, p); } else y = p;
          }
          // line integrals in metres of water-equivalent path; bone weighted up as a kV radiograph does
          float path = (0.15 * soft + 4.0 * bone) * dt;
          float g = pow(1.0 - exp(-9.0 * path), 0.85);
          outColor = vec4(g, clamp(pet, 0.0, 1.0), 0.0, 1.0);
        }`);
    this.scene = new THREE.Scene(); this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.sliceMat);
    this.scene.add(this.quad);
    const mkTarget = (size) => new THREE.WebGLRenderTarget(size, size, { depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    const mkPass = (size) => ({ size, target: mkTarget(size), ref: mkTarget(size), hasReference: false, captureNext: false });
    this.passes = { slice: mkPass(sliceSize), drr: mkPass(drrSize) };
    // Resolve: grey, or with a reference the reference in green under the live picture in magenta (as the surface DRR does);
    // PET in a hot colour scale on top when there is no reference.
    this.resolve = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false,
      uniforms: { uT: { value: null }, uRefT: { value: null }, uRef: { value: 0 }, uPx: { value: 0 }, uBlur: { value: 0 }, uPetMix: { value: 0 } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: `
        uniform sampler2D uT; uniform sampler2D uRefT; uniform float uRef; uniform float uPx; uniform float uBlur; uniform float uPetMix; varying vec2 vUv;
        vec2 soft(sampler2D t){
          vec2 c = texture2D(t, vUv).rg;
          if (uBlur < 0.5) return c;
          c *= 0.28;
          for (int i = 0; i < 4; i++) { float a = 1.5707963 * float(i); vec2 d = vec2(cos(a), sin(a)); c += texture2D(t, vUv + d * uPx).rg * 0.12; c += texture2D(t, vUv + vec2(d.x - d.y, d.x + d.y) * uPx).rg * 0.06; }
          return c;
        }
        void main(){
          vec2 v = soft(uT); float r = soft(uRefT).r;
          vec3 col = mix(vec3(v.r), vec3(v.r, r, v.r), uRef);
          float x = v.g;                                   // PET: black-red-yellow-white, more opaque the hotter
          vec3 hot = vec3(clamp(3.0 * x, 0.0, 1.0), clamp(3.0 * x - 1.0, 0.0, 1.0), clamp(3.0 * x - 2.0, 0.0, 1.0));
          col = mix(col, hot, uPetMix * (1.0 - uRef) * smoothstep(0.06, 0.45, x) * 0.8);
          gl_FragColor = vec4(col, 1.0);
        }` });
    this.resolveScene = new THREE.Scene();
    this.resolveScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolve));
  }

  setWindow(level, width) { this.uniforms.uWindow.value.set(level, width); }
  setPet(on) { this.petOn = !!on && this.hasPet; }

  // Keep the next rendered frame of each picture as its reference (green). The pictures must then be camera-fixed to mean anything:
  // the page renders the slice in the camera plane while a reference is held.
  captureReference() { for (const p of Object.values(this.passes)) p.captureNext = true; }
  clearReference(which = null) { for (const [k, p] of Object.entries(this.passes)) if (!which || k === which) { p.captureNext = false; p.hasReference = false; } }
  holdsReference(which) { const p = this.passes[which]; return p.hasReference || p.captureNext; }

  // Front-to-back position (raw frame z, front is -z) of the anchor bone in its current pose.
  anchorZ() {
    if (!this.anchor) return 0;
    const a = this.anchor.mesh.geometry.attributes.position.array;
    let z = 0;
    for (let i = 2; i < a.length; i += 3) z += a[i];
    return z / (a.length / 3);
  }

  _frames() {
    const u = this.uniforms, fr = this.layer.lastFrames;
    if (fr && fr.soft) {
      const n = Math.min(fr.soft.length, MAX_FRAMES);
      for (let k = 0; k < n; k++) {
        const f = fr.soft[k], R = f.R;
        u.uR.value[k].set(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);     // row-major in, so the shader's uR * v is R v
        u.uC.value[k].set(f.c[0], f.c[1], f.c[2]); u.uCp.value[k].set(f.cp[0], f.cp[1], f.cp[2]); u.uS.value[k] = f.s || 1;
      }
      u.uHasFrames.value = 1;
    }
    u.uHalf.value = this.half; u.uPetOn.value = this.petOn ? 1 : 0;
  }

  _draw(which, material, viewport, blur) {
    const { THREE, renderer } = this, pass = this.passes[which];
    const prev = renderer.getRenderTarget(), autoClear = renderer.autoClear;
    this.quad.material = material; renderer.autoClear = false;
    renderer.setRenderTarget(pass.target); renderer.render(this.scene, this.cam);
    if (pass.captureNext) {                                // same view, same pose: this frame becomes the reference
      renderer.setRenderTarget(pass.ref); renderer.render(this.scene, this.cam);
      pass.captureNext = false; pass.hasReference = true;
    }
    renderer.setRenderTarget(prev);
    const r = this.resolve.uniforms;
    r.uT.value = pass.target.texture; r.uRefT.value = pass.ref.texture; r.uRef.value = pass.hasReference ? 1 : 0;
    r.uPx.value = 1 / pass.size; r.uBlur.value = blur ? 1 : 0; r.uPetMix.value = this.petOn ? 1 : 0;
    if (viewport) { renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissorTest(true); }
    renderer.render(this.resolveScene, this.cam);
    renderer.autoClear = autoClear;
    if (viewport) renderer.setScissorTest(false);
  }

  // Slice. centre: [x, y] picture centre (raw MHR frame, metres); offsetM: slice position relative to the anchor bone (negative =
  // anterior). bodyAligned: cut a coronal plane of the rest body carried by the anchor vertebra's frame (centre is then ignored:
  // the picture is centred on the rest body); otherwise the plane is fixed to the camera, like the DRR.
  render(centre = [0, -1.3], offsetM = -0.05, viewport = null, bodyAligned = true) {
    const u = this.uniforms;
    this._frames();
    const k = this.anchor ? 1 + this.layer.atlas.chain.indexOf(this.anchor.s.name) : 0;      // soft frames: pelvis, then the chain
    if (bodyAligned && k > 0) {
      u.uPlaneFrame.value = k; u.uCentre.value.set(0, -1.3, 0); u.uZ.value = this.anchor.s.centroid[2] + offsetM;
      if (this.followCentre && centre && u.uHasFrames.value > 0.5) {
        // a zoomed picture is centred on a point of the POSED body: here the plane is drawn in rest coordinates, so that
        // point is taken back through the frame that carries the plane (rest = c + R^T (p - cp) / s)
        const THREE = this.THREE, R = u.uR.value[k].clone().transpose();
        const rest = new THREE.Vector3(centre[0], centre[1], centre.length > 2 && centre[2] ? centre[2] : this.anchorZ()).sub(u.uCp.value[k]).applyMatrix3(R)
          .multiplyScalar(1 / (u.uS.value[k] || 1)).add(u.uC.value[k]);
        u.uCentre.value.set(rest.x, rest.y, 0);
      }
    } else {
      u.uPlaneFrame.value = -1; u.uCentre.value.set(centre[0], centre[1], 0); u.uZ.value = this.anchorZ() + offsetM;
    }
    this._draw("slice", this.sliceMat, viewport, false);
  }

  // DRR through the CT voxels, seen like the surface DRR (xray.js): from the front, or turned by orbit { yaw, pitch } (radians).
  // basis: optional { back, up } in the raw frame, replacing orbit (see xray.js): a viewpoint fixed in the room.
  renderDrr(centre = [0, -1.3], viewport = null, orbit = null, basis = null) {
    const u = this.uniforms, yaw = orbit ? orbit.yaw : 0, pitch = orbit ? orbit.pitch : 0;
    this._frames();
    const cp = Math.cos(pitch), sp = Math.sin(pitch), THREE = this.THREE;
    const back = basis ? new THREE.Vector3(...basis.back) : new THREE.Vector3(Math.sin(yaw) * cp, sp, -Math.cos(yaw) * cp);   // from the centre to the viewpoint
    const right = (basis ? new THREE.Vector3(...basis.up) : new THREE.Vector3(0, -1, 0)).cross(back).normalize(), up = back.clone().cross(right);
    u.uRight.value.copy(right); u.uUp.value.copy(up); u.uFwd.value.copy(back).multiplyScalar(-1);
    // the same centre as the surface DRR (xray.js), so that the two coincide from every direction and can be blended
    u.uCentre.value.set(centre[0], centre[1], centre[2] || 0); u.uDepth.value = 0.6;
    this._draw("drr", this.drrMat, viewport, true);
  }
}
