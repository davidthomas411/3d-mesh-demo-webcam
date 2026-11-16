// RigStream on linear motion: extrapolation must beat holding the last mesh, and must not run past the horizon cap.
import { RigStream } from "../../anatomy/stream.js";
const n = 30, mk = t => Float32Array.from({ length: n }, (_, i) => (i % 3 === 0 ? 0.001 * t : 0));      // x moves 1 mm per ms
const hold = new RigStream(n, { horizonMs: 0, gain: 0, smoothMs: 0 }), ext = new RigStream(n, { horizonMs: 120, gain: 1, smoothMs: 0 });
for (const s of [hold, ext]) { s.push(mk(0), 0); s.push(mk(80), 80); }
const e = s => Math.abs(s.sample(160)[0] - 0.001 * 160);
console.log("hold error", e(hold), "extrapolated error", e(ext));
if (!(e(ext) < 1e-6 && e(hold) > 0.07)) { console.error("extrapolation did not track linear motion"); process.exit(1); }
const far = ext.sample(1000)[0];
if (far > 0.001 * (80 + 120) + 1e-6) { console.error("horizon cap not applied", far); process.exit(1); }
process.exit(0);
