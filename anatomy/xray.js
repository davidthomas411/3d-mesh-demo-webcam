// Real-time radiograph of the posed rig: every closed bone mesh is drawn twice into a float target, front faces adding -depth and back faces +depth, so each pixel
// accumulates the path length through bone along the view ray (sum of back minus front over all closed surfaces). A resolve pass maps thickness to grey like the offline
// CT-warp radiograph (water-equivalent cm, window 0-10, gamma 0.75). Open meshes (scapula, costal cartilage) cannot be integrated this way and add a fixed thin layer.
// Needs a WebGL2 context. The meshes are the AnatomyLayer's own geometries, so the view always shows the pose the rig just computed.
const OPEN = /^(scapula_|costal_cartilages)/;

export class XrayView {
  constructor(THREE, renderer, layer, { size = 512, density = 1.7, windowCm = 10, gamma = 0.75, openCm = 0.5, halfExtentM = 0.55 } = {}) {
    this.THREE = THREE; this.renderer = renderer; this.layer = layer; this.size = size; this.density = density; this.windowCm = windowCm; this.gamma = gamma; this.half = halfExtentM;
    this.scene = new THREE.Scene(); this.meshes = [];
    const mk = (side, sign, constant) => new THREE.ShaderMaterial({
      side, depthTest: false, depthWrite: false, transparent: true, blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      uniforms: { uSign: { value: sign }, uConst: { value: constant } },
      vertexShader: "varying float vD; void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vD = -mv.z; gl_Position = projectionMatrix * mv; }",
      fragmentShader: "uniform float uSign; uniform float uConst; varying float vD; void main(){ gl_FragColor = vec4(uConst > 0.0 ? uConst : uSign * vD, 0.0, 0.0, 1.0); }" });
    for (const e of layer.entries) {
      const s = e.s;
      if (s.bind === "soft" || s.name === "brain") continue;
      e.forceUpdate = true;                                                  // keep this geometry posed even when the layer's own mesh is hidden
      const open = OPEN.test(s.name), g = e.mesh.geometry;
      const specs = open ? [[THREE.DoubleSide, 1, openCm / 100]] : [[THREE.FrontSide, -1, 0], [THREE.BackSide, 1, 0]];
      for (const [side, sign, c] of specs) { const m = new THREE.Mesh(g, mk(side, sign, c)); m.frustumCulled = false; this.scene.add(m); this.meshes.push(m); }
    }
    this.target = new THREE.WebGLRenderTarget(size, size, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    this.cam = new THREE.OrthographicCamera(-this.half, this.half, this.half, -this.half, 0, 1.0);
    this.quadScene = new THREE.Scene(); this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.resolve = new THREE.ShaderMaterial({
      depthTest: false, depthWrite: false, uniforms: { uT: { value: this.target.texture }, uK: { value: density * 100 / windowCm }, uG: { value: gamma } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: "uniform sampler2D uT; uniform float uK; uniform float uG; varying vec2 vUv; void main(){ float t = max(texture2D(uT, vUv).r, 0.0); float v = pow(clamp(t * uK, 0.0, 1.0), uG); gl_FragColor = vec4(vec3(v), 1.0); }" });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolve));
    layer.xrayActive = true;
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
    renderer.setRenderTarget(prev); renderer.autoClear = autoClear; renderer.setClearColor(pc, pa);
    if (viewport) { renderer.setViewport(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissor(viewport.x, viewport.y, viewport.w, viewport.h); renderer.setScissorTest(true); }
    renderer.render(this.quadScene, this.quadCam);
    if (viewport) renderer.setScissorTest(false);
  }
}
