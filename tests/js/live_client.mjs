// End-to-end check of the live path without a browser: connect to a /stream server (replay_server.py or webcam_server.py), send frames the way webcam.html does, feed the
// meshes through RigStream and run the full rig (physiologic spine, joints, scapular contact, cartilage; every structure) at 60 Hz. Reports mesh rate, capture-to-arrival latency
// and the rig's per-frame cost.   usage: node live_client.mjs ws://host:port/stream ATLAS_PREFIX SECONDS
import fs from "fs";
import { parseAtlas, computeFrames, transformStructure } from "../../anatomy/anatomy.js";
import { RigStream } from "../../anatomy/stream.js";
const [url, prefix, secS = "10"] = process.argv.slice(2), SECONDS = +secS, NV = 18439 * 3;
const bin = fs.readFileSync(prefix + ".bin");
const atlas = parseAtlas(JSON.parse(fs.readFileSync(prefix + ".json", "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const out = atlas.structures.map(s => new Float32Array(s.pos.length)), stream = new RigStream(NV, { horizonMs: 120, gain: 1, smoothMs: 0 });
const lat = [], infer = [], rigMs = [], gaps = []; let meshes = 0, expectBinary = false, meta = null, lastTick = 0, frames = 0;
const ws = new WebSocket(url); ws.binaryType = "arraybuffer";
const sendFrame = () => { ws.send(JSON.stringify({ capture_timestamp_ms: performance.now(), encode_ms: 0 })); ws.send(new Uint8Array(300)); };
ws.onopen = () => { sendFrame(); };
ws.onmessage = ev => {
  if (typeof ev.data === "string") { meta = JSON.parse(ev.data); if (meta.error) { console.error(meta.error); process.exit(2); } return; }
  const arrive = performance.now(), v = new Float32Array(ev.data);
  if (v.length !== NV) { console.error("unexpected payload", v.length); process.exit(2); }
  lat.push(arrive - Number(meta.capture_timestamp_ms)); infer.push(meta.inference_ms); meshes++;
  stream.push(v, Number(meta.capture_timestamp_ms)); sendFrame();
};
ws.onerror = e => { console.error("websocket error", e.message || e); process.exit(2); };
const t0 = performance.now();
function tick() {
  const now = performance.now();
  if (now - t0 > SECONDS * 1000) return finish();
  const skin = stream.sample(now);
  if (skin) {
    const a = performance.now(), fr = computeFrames(atlas, skin, 0, 1, "physio");
    for (let i = 0; i < atlas.structures.length; i++) transformStructure(atlas.structures[i], fr, out[i], null, null, 1);
    rigMs.push(performance.now() - a); frames++; if (lastTick) gaps.push(now - lastTick); lastTick = now;
  }
  setTimeout(tick, Math.max(0, 1000 / 60 - (performance.now() - now)));
}
const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(s.length * q))]; };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
function finish() {
  console.log(`meshes received: ${meshes} (${(meshes / SECONDS).toFixed(1)} Hz) | estimator latency (server-side) mean ${mean(infer).toFixed(0)} ms`);
  console.log(`capture -> mesh arrives in the client: mean ${mean(lat).toFixed(0)} ms, p95 ${pct(lat, 0.95).toFixed(0)} ms`);
  console.log(`rig per display frame: mean ${mean(rigMs).toFixed(2)} ms, p95 ${pct(rigMs, 0.95).toFixed(2)} ms, max ${Math.max(...rigMs).toFixed(2)} ms | display frames ${frames} (${(frames / SECONDS).toFixed(0)} fps), frame gaps p95 ${pct(gaps, 0.95).toFixed(1)} ms`);
  ws.close(); process.exit(0);
}
setTimeout(tick, 500);
