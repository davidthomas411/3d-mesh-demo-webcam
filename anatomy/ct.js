// Real-time CT-style coronal slice of the posed rig. NOT a CT: there are no voxels in the atlas, only labelled surfaces, so each
// tissue gets one flat grey (a "label CT"). It answers "what would a coronal cut through the posed anatomy look like", not "what
// are the Hounsfield units here". A real CT reformat needs the subject's volume and a GPU warp (docs/PATIENT_RIG_PLAN.md).
//
// How: an orthographic camera looks from the front with its near plane ON the slice plane. For a closed surface, the number of back
// faces minus the number of front faces met by a ray starting on the plane is 1 if the start point is inside the surface and 0 if
// not. Every closed mesh is therefore drawn twice with additive blending into a float target (front faces add -g, back faces +g),
// where g is that tissue's grey RELATIVE to what surrounds it: the skin adds soft tissue over air, a lung subtracts from soft
// tissue, a bone adds to it. One pass, no stencil, and it uses the AnatomyLayer's own geometries, so it shows the pose the rig just
// computed. Open meshes (scapula, costal cartilage) cannot be sliced this way and are left out. Needs WebGL2 (float targets).
const OPEN = /^(scapula_|costal_cartilages)/;
const BONE_GROUPS = new Set(["skull", "spine", "ribcage", "shoulder", "pelvis_leg"]);
const SOFT = 0.42;                                         // grey of unlabelled soft tissue inside the skin (air = 0)
// grey of a structure relative to soft tissue, by atlas group (by name where a group is mixed)
const GROUP_DELTA = { lung: -0.33, airway: -0.38, gi: -0.05, heart: 0.07, artery: 0.1, vein: 0.07, urinary: 0.05, muscle: 0.04, neuro_endo: 0.03, other: 0.05 };
const BONE_DELTA = 0.5;

function greyDelta(s) {
  if (BONE_GROUPS.has(s.group)) return BONE_DELTA;
  if (/^(liver|spleen|pancreas)/.test(s.name)) return 0.08;
  if (/^(stomach|colon|small_bowel|duodenum|esophagus)/.test(s.name)) return -0.05;
  return GROUP_DELTA[s.group] ?? 0.04;
}

export class CtSliceView {
  // skinGeometry: the live body mesh geometry (raw MHR frame, same as the layer's geometries).
  constructor(THREE, renderer, layer, skinGeometry, { size = 512, halfExtentM = 0.55, anchor = "vertebrae_T8" } = {}) {
    this.THREE = THREE; this.renderer = renderer; this.layer = layer; this.size = size; this.half = halfExtentM;
    this.scene = new THREE.Scene(); this.entries = [];
    this.anchor = layer.byName.get(anchor) || null;        // the slice plane follows this bone front-to-back
    const mk = (side, value) => new THREE.ShaderMaterial({
      side, depthTest: false, depthWrite: false, transparent: true, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      uniforms: { uV: { value } },
      vertexShader: "void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: "uniform float uV; void main(){ gl_FragColor = vec4(uV, 0.0, 0.0, 1.0); }" });
    const add = (geometry, grey) => {
      // a ray leaving a closed surface crosses a face seen from behind: back faces +grey, front faces -grey
      const mats = [];
      for (const [side, v] of [[THREE.BackSide, grey], [THREE.FrontSide, -grey]]) {
        const m = new THREE.Mesh(geometry, mk(side, v)); m.frustumCulled = false; this.scene.add(m); mats.push(m.material);
      }
      return mats;
    };
    this.skinGeometry = skinGeometry; this.skinMats = add(skinGeometry, SOFT); this.skinWindingKnown = false;
    for (const e of layer.entries) {
      if (OPEN.test(e.s.name)) continue;
      this.entries.push(e);
      add(e.mesh.geometry, greyDelta(e.s));
    }
    this.target = new THREE.WebGLRenderTarget(size, size, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.cam = new THREE.OrthographicCamera(-this.half, this.half, this.half, -this.half, 0, 2.0);
    this.quadScene = new THREE.Scene(); this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.resolve = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false, uniforms: { uT: { value: this.target.texture } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: "uniform sampler2D uT; varying vec2 vUv; void main(){ float v = clamp(texture2D(uT, vUv).r, 0.0, 1.0); gl_FragColor = vec4(vec3(v), 1.0); }" });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolve));
    this.setActive(true);
  }

  // While active, every sliced structure is posed each frame even if its 3-D mesh is hidden.
  setActive(on) {
    for (const e of this.entries) e.ctForce = on;
    this.layer.ctActive = on;
  }

  // Front-to-back position (raw frame z, front is -z) of the anchor bone in its current pose.
  anchorZ() {
    if (!this.anchor) return 0;
    const a = this.anchor.mesh.geometry.attributes.position.array;
    let z = 0;
    for (let i = 2; i < a.length; i += 3) z += a[i];
    return z / (a.length / 3);
  }

  // centre: [x, y] of the picture centre (raw MHR frame, metres). offsetM: slice position relative to the anchor bone, negative = anterior.
  render(centre = [0, -1.3], offsetM = -0.05, viewport = null) {
    const { THREE, renderer, cam } = this, z0 = this.anchorZ() + offsetM;
    if (!this.skinWindingKnown) {                          // the skin's triangle winding is only known once a mesh has arrived
      const v = signedVolume(this.skinGeometry);
      if (Math.abs(v) > 1e-6) { const s = v > 0 ? 1 : -1; this.skinMats[0].uniforms.uV.value = s * SOFT; this.skinMats[1].uniforms.uV.value = -s * SOFT; this.skinWindingKnown = true; }
    }
    cam.up.set(0, -1, 0);                                  // MHR y points down
    cam.left = -this.half; cam.right = this.half; cam.top = this.half; cam.bottom = -this.half;      // the extent may change (zoom)
    cam.position.set(centre[0], centre[1], z0); cam.lookAt(centre[0], centre[1], z0 + 1);   // from the front, looking back; near plane = slice plane
    cam.updateProjectionMatrix(); cam.updateMatrixWorld();
    const prev = renderer.getRenderTarget(), pc = renderer.getClearColor(new THREE.Color()), pa = renderer.getClearAlpha(), autoClear = renderer.autoClear;
    renderer.setRenderTarget(this.target); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.autoClear = false;
    renderer.render(this.scene, cam);
    renderer.setRenderTarget(prev); renderer.autoClear = autoClear; renderer.setClearColor(pc, pa);
    if (viewport) { renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissorTest(true); }
    renderer.render(this.quadScene, this.quadCam);
    if (viewport) renderer.setScissorTest(false);
  }
}

// > 0 when the triangles of a closed mesh wind counter-clockwise seen from outside.
function signedVolume(geometry) {
  const p = geometry.attributes.position.array, ix = geometry.index.array;
  let v = 0;
  for (let i = 0; i < ix.length; i += 3) {
    const a = ix[i] * 3, b = ix[i + 1] * 3, c = ix[i + 2] * 3;
    v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return v;
}
