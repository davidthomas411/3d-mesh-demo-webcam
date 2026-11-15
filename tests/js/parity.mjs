// Node parity check: anatomy.js weightedRigid vs the numpy reference (reads a JSON case).
import { weightedRigid } from "../../anatomy/anatomy.js";
import fs from "fs";
const c = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const { R, cs, cd } = weightedRigid(Float32Array.from(c.sRest.flat()), Uint32Array.from(c.sIdx),
  Float32Array.from(c.sW), Float32Array.from(c.live.flat()));
const err = Math.max(...R.map((v, i) => Math.abs(v - c.R[i])));
console.log(JSON.stringify({ maxErrR: err, cs, cd }));
process.exit(err < 1e-4 ? 0 : 1);
