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
    // the path through it would make the whole head solid bone. Each crossing of its surface adds the wall thickness instead,
    // longer where the ray grazes the wall, which is what gives a real skull its bright rim.
    const shell = (mm) => new THREE.ShaderMaterial({
      side: THREE.DoubleSide, depthTest: false, depthWrite: false, transparent: true, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      uniforms: { uT: { value: mm / 1000 } },
      vertexShader: "varying vec3 vN; void main(){ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: "uniform float uT; varying vec3 vN; void main(){ gl_FragColor = vec4(uT / max(abs(normalize(vN).z), 0.22), 0.0, 0.0, 1.0); }" });
    const sources = [[layer, null], ...extra];
    for (const [src, keep] of sources) for (const e of src.entries) {
      const s = e.s;
      if (s.bind === "soft" || s.name === "brain" || (keep && !keep(s))) continue;
      e.forceUpdate = true;                                                  // keep this geometry posed even when the layer's own mesh is hidden
      const open = OPEN.test(s.name), g = e.mesh.geometry;
      if (s.shell_mm) { const m = new THREE.Mesh(g, shell(s.shell_mm)); m.frustumCulled = false; this.scene.add(m); this.meshes.push(m); continue; }
      const specs = open ? [[THREE.DoubleSide, 1, openCm / 100]] : [[THREE.FrontSide, -1, 0], [THREE.BackSide, 1, 0]];
      for (const [side, sign, c] of specs) { const m = new THREE.Mesh(g, mk(side, sign, c)); m.frustumCulled = false; this.scene.add(m); this.meshes.push(m); }
    }
    const mkTarget = () => new THREE.WebGLRenderTarget(size, size, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.target = mkTarget();
    this.refTarget = mkTarget();                                             // bone thickness at the moment a reference was captured
    this.hasReference = false; this._captureNext = false;
    this.cam = new THREE.OrthographicCamera(-this.half, this.half, this.half, -this.half, 0, 1.0);
    this.quadScene = new THREE.Scene(); this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.resolve = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false, uniforms: { uT: { value: this.target.texture }, uR: { value: this.refTarget.texture }, uRef: { value: 0 }, uK: { value: density * 100 / windowCm }, uG: { value: gamma } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      // With a reference: reference in green, live in magenta. Where the two agree the pixel is grey; a shift shows as a green
      // edge where the bone was and a magenta edge where it is now.
      fragmentShader: "uniform sampler2D uT; uniform sampler2D uR; uniform float uRef; uniform float uK; uniform float uG; varying vec2 vUv; void main(){ float v = pow(clamp(max(texture2D(uT, vUv).r, 0.0) * uK, 0.0, 1.0), uG); float r = pow(clamp(max(texture2D(uR, vUv).r, 0.0) * uK, 0.0, 1.0), uG); gl_FragColor = vec4(mix(vec3(v), vec3(v, r, v), uRef), 1.0); }" });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolve));
    for (const [src] of sources) src.xrayActive = true;
  }
  // view: "ap" (from the front) or "lateral" (from the person's left, face to the left of the image). centre: [x, y, z] of the picture centre in the raw MHR frame (metres).
  render(view = "ap", centre = [0, -1.3, 0], viewport = null) {
    const { THREE, renderer, cam } = this, c = centre, d = 0.6;
    cam.up.set(0, -1, 0);                                                    // MHR y points down
    if (view === "ap") { cam.position.set(c[0], c[1], c[2] - d); cam.lookAt(c[0], c[1], c[2]); cam.left = -this.half; cam.right = this.half; }
    else { cam.position.set(c[0] + d, c[1], c[2]); cam.lookAt(c[0], c[1], c[2]); cam.left = -this.half; cam.right = this.half; }
    cam.near = 0; cam.far = 2 * d; cam.updateProjectionMatrix(); cam.updateMatrixWorld();
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
