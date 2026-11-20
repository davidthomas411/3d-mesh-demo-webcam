// Real-time radiograph of the posed rig: every closed bone mesh is drawn twice into a float target, front faces adding -depth and back faces +depth, so each pixel
// accumulates the path length through bone along the view ray (sum of back minus front over all closed surfaces). A resolve pass maps thickness to grey like the offline
// CT-warp radiograph (water-equivalent cm, window 0-10, gamma 0.75). Open meshes (scapula, costal cartilage) cannot be integrated this way and add a fixed thin layer.
// Needs a WebGL2 context. The meshes are the AnatomyLayer's own geometries, so the view always shows the pose the rig just computed.
const OPEN = /^(scapula_|costal_cartilages)/;

export class XrayView {
  // extra: more bones from other layers, as [[layer, (structure) => boolean], ...] (e.g. the limb bones another atlas has)
  constructor(THREE, renderer, layer, { size = 512, density = 1.7, windowCm = 10, gamma = 0.75, openCm = 0.5, halfExtentM = 0.55, extra = [] } = {}) {
    this.THREE = THREE; this.renderer = renderer; this.layer = layer; this.size = size; this.density = density; this.windowCm = windowCm; this.gamma = gamma; this.half = halfExtentM;
    this.scene = new THREE.Scene(); this.meshes = [];
    const mk = (side, sign, constant) => new THREE.ShaderMaterial({
      side, depthTest: false, depthWrite: false, transparent: true, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      uniforms: { uSign: { value: sign }, uConst: { value: constant } },
      vertexShader: "varying float vD; void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vD = -mv.z; gl_Position = projectionMatrix * mv; }",
      fragmentShader: "uniform float uSign; uniform float uConst; varying float vD; void main(){ gl_FragColor = vec4(uConst > 0.0 ? uConst : uSign * vD, 0.0, 0.0, 1.0); }" });
    // A structure marked shell_mm is a closed surface standing for a hollow bone (SKEL's skull has no cranial cavity): integrating
    // the path through it would make the whole head solid bone. Its cavity is therefore subtracted: the same surface, shrunk
    // about its centre by the wall thickness, is drawn with the opposite sign, so each ray gets the path through the wall only.
    // That is a real path length, so the skull is as soft as the other bones (a fixed thickness per crossing looked jagged).
    const hollow = (side, sign) => new THREE.ShaderMaterial({
      side, depthTest: false, depthWrite: false, transparent: true, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      uniforms: { uSign: { value: sign }, uCentre: { value: new THREE.Vector3() }, uScale: { value: 1 } },
      vertexShader: "uniform vec3 uCentre; uniform float uScale; varying float vD; void main(){ vec4 mv = modelViewMatrix * vec4(uCentre + uScale * (position - uCentre), 1.0); vD = -mv.z; gl_Position = projectionMatrix * mv; }",
      fragmentShader: "uniform float uSign; varying float vD; void main(){ gl_FragColor = vec4(uSign * vD, 0.0, 0.0, 1.0); }" });
    this.shells = [];
    const sources = [[layer, null], ...extra];
    for (const [src, keep] of sources) for (const e of src.entries) {
      const s = e.s;
      if (s.bind === "soft" || s.name === "brain" || (keep && !keep(s))) continue;
      e.forceUpdate = true;                                                  // keep this geometry posed even when the layer's own mesh is hidden
      const open = OPEN.test(s.name), g = e.mesh.geometry;
      if (s.shell_mm) {
        // rest size of the bone -> how much to shrink the copy that stands for the cavity
        let r = 0; const c0 = s.centroid, p0 = s.pos;
        for (let i = 0; i < p0.length; i += 3) r += Math.hypot(p0[i] - c0[0], p0[i + 1] - c0[1], p0[i + 2] - c0[2]);
        r /= p0.length / 3;
        const inner = [hollow(THREE.FrontSide, 1), hollow(THREE.BackSide, -1)];
        for (const m of inner) m.uniforms.uScale.value = Math.max(0.5, 1 - (s.shell_mm / 1000) / r);
        for (const mat of [mk(THREE.FrontSide, -1, 0), mk(THREE.BackSide, 1, 0), ...inner]) { const m = new THREE.Mesh(g, mat); m.frustumCulled = false; this.scene.add(m); this.meshes.push(m); }
        this.shells.push({ g, inner });
        continue;
      }
      const gain = s.xray_gain ?? 1;                                           // < 1 for small dense pieces that stack along a ray (teeth)
      const specs = open ? [[THREE.DoubleSide, 1, gain * openCm / 100]] : [[THREE.FrontSide, -gain, 0], [THREE.BackSide, gain, 0]];
      for (const [side, sign, c] of specs) { const m = new THREE.Mesh(g, mk(side, sign, c)); m.frustumCulled = false; this.scene.add(m); this.meshes.push(m); }
    }
    const mkTarget = () => new THREE.WebGLRenderTarget(size, size, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.target = mkTarget();
    this.refTarget = mkTarget();                                             // bone thickness at the moment a reference was captured
    this.hasReference = false; this._captureNext = false;
    this.cam = new THREE.OrthographicCamera(-this.half, this.half, this.half, -this.half, 0, 1.0);
    this.quadScene = new THREE.Scene(); this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.resolve = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false, uniforms: { uT: { value: this.target.texture }, uR: { value: this.refTarget.texture }, uRef: { value: 0 }, uK: { value: density * 100 / windowCm }, uG: { value: gamma }, uPx: { value: 1 / size } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      // With a reference: reference in green, live in magenta. Where the two agree the pixel is grey; a shift shows as a green
      // edge where the bone was and a magenta edge where it is now.
      // The thickness image has hard polygon edges (no multisampling on a float target): it is read through a 13-tap blur, which
      // is also closer to a radiograph's own blur.
      fragmentShader: "uniform sampler2D uT; uniform sampler2D uR; uniform float uRef; uniform float uK; uniform float uG; uniform float uPx; varying vec2 vUv; "
        + "float soft(sampler2D t){ float c = max(texture2D(t, vUv).r, 0.0) * 0.20; "
        + "for (int i = 0; i < 4; i++) { float a = 1.5707963 * float(i); vec2 d = vec2(cos(a), sin(a)); c += max(texture2D(t, vUv + d * uPx).r, 0.0) * 0.12; c += max(texture2D(t, vUv + vec2(d.x - d.y, d.x + d.y) * uPx).r, 0.0) * 0.05; c += max(texture2D(t, vUv + d * 2.0 * uPx).r, 0.0) * 0.03; } return c; } "
        // brightness follows attenuation, 1 - exp(-mu t), as in a radiograph: thick bone approaches white smoothly instead of
        // clipping to a flat white patch (the solid facial block of a generic skull did)
        + "float tone(float t){ return pow(1.0 - exp(-1.6 * t * uK), uG); } "
        + "void main(){ float v = tone(soft(uT)); float r = tone(soft(uR)); gl_FragColor = vec4(mix(vec3(v), vec3(v, r, v), uRef), 1.0); }" });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolve));
    for (const [src] of sources) src.xrayActive = true;
  }
  // view: "ap" (from the front) or "lateral" (from the person's left, face to the left of the image). centre: [x, y, z] of the picture centre in the raw MHR frame (metres).
  // orbit: optional { yaw, pitch } in radians, turning the viewpoint about the centre (0, 0 = the "ap" view).
  render(view = "ap", centre = [0, -1.3, 0], viewport = null, orbit = null) {
    const { THREE, renderer, cam } = this, c = centre, d = 0.6;
    cam.up.set(0, -1, 0);                                                    // MHR y points down
    if (orbit) {
      const cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
      cam.position.set(c[0] + d * Math.sin(orbit.yaw) * cp, c[1] + d * sp, c[2] - d * Math.cos(orbit.yaw) * cp);
      cam.lookAt(c[0], c[1], c[2]); cam.left = -this.half; cam.right = this.half;
    } else if (view === "ap") { cam.position.set(c[0], c[1], c[2] - d); cam.lookAt(c[0], c[1], c[2]); cam.left = -this.half; cam.right = this.half; }
    else { cam.position.set(c[0] + d, c[1], c[2]); cam.lookAt(c[0], c[1], c[2]); cam.left = -this.half; cam.right = this.half; }
    cam.near = 0; cam.far = 2 * d; cam.updateProjectionMatrix(); cam.updateMatrixWorld();
    for (const sh of this.shells) {                                          // centre of each hollow bone in its current pose
      const a = sh.g.attributes.position.array; let x = 0, y = 0, z = 0, n = 0;
      for (let i = 0; i < a.length; i += 30) { x += a[i]; y += a[i + 1]; z += a[i + 2]; n++; }
      for (const m of sh.inner) m.uniforms.uCentre.value.set(x / n, y / n, z / n);
    }
    // depth along the view ray measured from the camera plane; thickness is a difference of depths, so the origin does not matter, but keep it near the body for precision
    const prev = renderer.getRenderTarget(), pc = renderer.getClearColor(new THREE.Color()), pa = renderer.getClearAlpha(), autoClear = renderer.autoClear;
    renderer.setRenderTarget(this.target); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.autoClear = false;
    renderer.render(this.scene, cam);
    if (this._captureNext) {                                                 // same view, same pose: this frame becomes the reference
      renderer.setRenderTarget(this.refTarget); renderer.clear(); renderer.render(this.scene, cam);
      this._captureNext = false; this.hasReference = true; this.resolve.uniforms.uRef.value = 1;
    }
    renderer.setRenderTarget(prev); renderer.autoClear = autoClear; renderer.setClearColor(pc, pa);
    if (viewport) { renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissorTest(true); }
    renderer.render(this.quadScene, this.quadCam);
    if (viewport) renderer.setScissorTest(false);
  }
  // Keep the next rendered frame as the reference radiograph (the view is fixed to the camera, so it records where the bones were).
  captureReference() { this._captureNext = true; }
  clearReference() { this._captureNext = false; this.hasReference = false; this.resolve.uniforms.uRef.value = 0; }
}
