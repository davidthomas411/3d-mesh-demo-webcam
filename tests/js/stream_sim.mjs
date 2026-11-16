// Simulate a slow, late estimator and measure how far the displayed mesh is from the person's true pose at display time.
// usage: node stream_sim.mjs SKINS30.bin N_FRAMES [estimatorHz=12] [latencyMs=120]
import fs from "fs";
import { RigStream } from "../../anatomy/stream.js";
const [path, nf, hzS = "12", latS = "120"] = process.argv.slice(2), NF = +nf, NV = 18439 * 3, HZ = +hzS, LAT = +latS;
const buf = fs.readFileSync(path), all = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const truthAt = tMs => { const f = tMs / (1000 / 30), i = Math.min(Math.max(Math.floor(f), 0), NF - 2), u = Math.min(Math.max(f - i, 0), 1); return { i, u }; };
function errMm(shown, tMs) {                                    // mean per-vertex distance to the true skin, mm
  const { i, u } = truthAt(tMs); let s = 0; const nv = NV / 3;
  for (let v = 0; v < nv; v++) { let d2 = 0; for (let k = 0; k < 3; k++) { const j = v * 3 + k, t = (1 - u) * all[i * NV + j] + u * all[(i + 1) * NV + j]; d2 += (shown[j] - t) ** 2; } s += Math.sqrt(d2); }
  return 1000 * s / nv;
}
const duration = (NF - 1) * 1000 / 30, configs = { "hold last": { horizonMs: 0, gain: 0, smoothMs: 0 }, "extrapolate 80 ms": { horizonMs: 80, gain: 1, smoothMs: 0 },
  "extrapolate 120 ms": { horizonMs: 120, gain: 1, smoothMs: 0 }, "extrapolate 120 ms, gain 0.8": { horizonMs: 120, gain: 0.8, smoothMs: 0 }, "extrapolate 160 ms, gain 0.8": { horizonMs: 160, gain: 0.8, smoothMs: 0 },
  "extrapolate 120 ms, gain 0.8 + smoothing 40 ms": { horizonMs: 120, gain: 0.8, smoothMs: 40 } };
console.log(`estimator ${HZ} Hz, latency ${LAT} ms, display 60 Hz, ${NF} frames of true motion (${(duration / 1000).toFixed(1)} s)`);
for (const [name, cfg] of Object.entries(configs)) {
  const rs = new RigStream(NV, cfg), errs = []; let nextArrival = LAT + 0, nextCapture = 0;
  const arrivals = []; for (let c = 0; c * 1000 / HZ < duration - LAT; c++) arrivals.push({ capture: c * 1000 / HZ, arrive: c * 1000 / HZ + LAT });
  let ai = 0; const t0 = performance.now();
  for (let now = LAT * 2; now < duration; now += 1000 / 60) {
    while (ai < arrivals.length && arrivals[ai].arrive <= now) { const { i, u } = truthAt(arrivals[ai].capture); const f = new Float32Array(NV); for (let j = 0; j < NV; j++) f[j] = (1 - u) * all[i * NV + j] + u * all[(i + 1) * NV + j]; rs.push(f, arrivals[ai].capture); ai++; }
    const shown = rs.sample(now); if (shown) errs.push(errMm(shown, now));
  }
  errs.sort((a, b) => a - b); const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
  console.log(`${name.padEnd(48)} mean ${mean.toFixed(1)} mm   p95 ${errs[Math.floor(errs.length * 0.95)].toFixed(1)} mm   max ${errs[errs.length - 1].toFixed(1)} mm`);
}
