// Node parity check of the browser kinematics (anatomy.js) against the numpy reference, no three.js needed.
// usage: node parity_pose.mjs ATLAS_PREFIX CASE.json   (CASE: {skin, breath, depth, expected} with float32 .bin paths)
import fs from "fs";
import { parseAtlas, computeFrames, transformStructure } from "../../anatomy/anatomy.js";

const [prefix, casePath] = process.argv.slice(2);
const c = JSON.parse(fs.readFileSync(casePath, "utf8"));
const bin = fs.readFileSync(prefix + ".bin");
const atlas = parseAtlas(JSON.parse(fs.readFileSync(prefix + ".json", "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const sk = fs.readFileSync(c.skin), skin = new Float32Array(sk.buffer.slice(sk.byteOffset, sk.byteOffset + sk.byteLength));
const ex = fs.readFileSync(c.expected), exp = new Float32Array(ex.buffer.slice(ex.byteOffset, ex.byteOffset + ex.byteLength));
const fr = computeFrames(atlas, skin, c.breath, c.depth, c.spine || "skin");
let off = 0, worst = 0, worstName = "";
for (const s of atlas.structures) {
  const P = new Float64Array(s.pos.length);
  transformStructure(s, fr, P, null, null, 1);
  for (let i = 0; i < P.length; i++) { const e = Math.abs(P[i] - exp[off + i]); if (e > worst) { worst = e; worstName = s.name; } }
  off += P.length;
}
console.log(JSON.stringify({ maxErrM: worst, worstStructure: worstName, structures: atlas.structures.length }));
process.exit(worst < 2e-4 ? 0 : 1);
