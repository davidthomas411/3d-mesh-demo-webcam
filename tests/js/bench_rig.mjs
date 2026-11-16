// Time the browser rig (anatomy.js) in node: computeFrames, then transformStructure per structure class, over N runs with a changing skin.
// usage: node bench_rig.mjs ATLAS_PREFIX SKINS.bin N_FRAMES [spine=physio|skin] [supportStride=1]   (SKINS.bin: N_FRAMES x 18439 x 3 float32, rest-frame MHR skin vertices)
import fs from "fs";
import { parseAtlas, computeFrames, transformStructure } from "../../anatomy/anatomy.js";
const [prefix, skinPath, nFrames, spine = "physio", stride = "1"] = process.argv.slice(2), NF = +nFrames, NV = 18439;
const bin = fs.readFileSync(prefix + ".bin");
const atlas = parseAtlas(JSON.parse(fs.readFileSync(prefix + ".json", "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const sk = fs.readFileSync(skinPath), all = new Float32Array(sk.buffer.slice(sk.byteOffset, sk.byteOffset + sk.byteLength));
const bufs = atlas.structures.map(s => new Float32Array(s.pos.length)), nrm = atlas.structures.map(s => new Float32Array(s.pos.length));
const groups = { soft: [], rigid: [] }; atlas.structures.forEach((s, i) => (s.bind === "soft" ? groups.soft : groups.rigid).push(i));
const nv = idx => idx.reduce((a, i) => a + atlas.structures[i].pos.length / 3, 0);
const t = { frames: 0, soft: 0, rigid: 0 }, per = new Float64Array(NF);
for (let rep = 0; rep < 3; rep++) for (let f = 0; f < NF; f++) {                       // first pass warms the JIT; times from the last pass
  const skin = all.subarray(f * NV * 3, (f + 1) * NV * 3);
  const t0 = performance.now(); const fr = computeFrames(atlas, skin, 0, 1, spine, +stride); const t1 = performance.now();
  for (const i of groups.soft) transformStructure(atlas.structures[i], fr, bufs[i], null, null, 1); const t2 = performance.now();
  for (const i of groups.rigid) transformStructure(atlas.structures[i], fr, bufs[i], null, null, 1); const t3 = performance.now();
  if (rep === 2) { t.frames += t1 - t0; t.soft += t2 - t1; t.rigid += t3 - t2; per[f] = t3 - t0; }
}
const ms = x => (x / NF).toFixed(2);
console.log(`frames: ${ms(t.frames)} ms | soft tissue (${groups.soft.length} structures, ${nv(groups.soft)} vertices): ${ms(t.soft)} ms | bones (${groups.rigid.length}, ${nv(groups.rigid)} vertices): ${ms(t.rigid)} ms | total ${ms(t.frames + t.soft + t.rigid)} ms/frame (max ${Math.max(...per).toFixed(2)})`);
