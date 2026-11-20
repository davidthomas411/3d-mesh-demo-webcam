// Display-rate pose for a slow estimator. The model (Fast SAM 3D Body) delivers a skin mesh at about 10-13 Hz, 100-170 ms after the camera saw the person; the
// display runs at 60 Hz. RigStream keeps the last two meshes (with their CAPTURE timestamps) and, for any display time, estimates the mesh the person has NOW by
// extrapolating the vertex velocity, capped (horizonMs) and damped so a pose glitch cannot fling the body away. The rig (computeFrames / transformStructure) then runs
// on the estimate at display rate. No three.js dependency.
export class RigStream {
  constructor(nValues, { horizonMs = 120, gain = 1.0, smoothMs = 0 } = {}) {
    this.n = nValues; this.horizonMs = horizonMs; this.gain = gain; this.smoothMs = smoothMs;
    this.a = null; this.b = null; this.ta = 0; this.tb = 0; this.est = new Float32Array(nValues); this.shown = new Float32Array(nValues); this.hasShown = false; this.lastNow = 0;
  }
  // skin: Float32Array(nValues) in the atlas frame; tCapture: capture time of the frame in ms (client clock). Copies the data.
  push(skin, tCapture) {
    if (this.b && tCapture <= this.tb) return;                 // out-of-order frame
    this.a = this.b; this.ta = this.tb;
    this.b = this.a ? this.a === this.b ? new Float32Array(this.n) : this._spare() : new Float32Array(this.n);
    this.b.set(skin); this.tb = tCapture;
  }
  _spare() { return new Float32Array(this.n); }
  // Estimate of the mesh at display time nowMs (ms, same clock as tCapture). Returns a Float32Array that is reused between calls.
  sample(nowMs) {
    if (!this.b) return null;
    const est = this.est, b = this.b;
    if (!this.a) { est.set(b); }
    else {
      const dt = this.tb - this.ta, ahead = Math.min(Math.max(nowMs - this.tb, 0), this.horizonMs);
      const k = dt > 1 ? this.gain * ahead / dt : 0;           // fraction of the last inter-frame step to extrapolate
      const a = this.a;
      for (let i = 0; i < this.n; i++) est[i] = b[i] + k * (b[i] - a[i]);
    }
    // light exponential smoothing at display rate (removes the stair-step when a new mesh arrives)
    const dtS = this.hasShown ? Math.max(nowMs - this.lastNow, 0) : 0, alpha = this.smoothMs > 0 ? 1 - Math.exp(-dtS / this.smoothMs) : 1;
    if (!this.hasShown) { this.shown.set(est); this.hasShown = true; } else { const sh = this.shown; for (let i = 0; i < this.n; i++) sh[i] += alpha * (est[i] - sh[i]); }
    this.lastNow = nowMs;
    return this.shown;
  }
}
