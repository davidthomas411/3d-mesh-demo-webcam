// Headless check of anatomy/xray.js: serves the repo plus three.js (THREE_DIR) and a skin file, loads tests/xray_page.html in Chromium (software GL) and saves a screenshot.
// usage: THREE_DIR=/path/to/node_modules/three node xray_shot.mjs SKINS.bin FRAME OUT.png
import http from "http"; import fs from "fs"; import path from "path"; import { createRequire } from "module";
const require = createRequire(import.meta.url); let chromium; try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require("/opt/node22/lib/node_modules/playwright")); }
const [skins, frame, out] = process.argv.slice(2), ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), ".."), THREE = process.env.THREE_DIR;
const types = { ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };
const srv = http.createServer((req, res) => {
  const u = new URL(req.url, "http://x"); let f;
  if (u.pathname === "/skins.bin") f = skins; else if (u.pathname.startsWith("/three/")) f = path.join(THREE, u.pathname.slice(7)); else if (u.pathname === "/") f = path.join(ROOT, "tests/xray_page.html"); else f = path.join(ROOT, u.pathname);
  fs.readFile(f, (e, d) => { if (e) { res.writeHead(404); res.end(); } else { res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); res.end(d); } });
}).listen(0, async () => {
  const port = srv.address().port, b = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  const pg = await b.newPage({ viewport: { width: 1100, height: 640 } }); pg.on("console", m => { if (m.type() === "error") console.log("page error:", m.text()); }); pg.on("pageerror", e => console.log("page exception:", e.message));
  await pg.goto(`http://localhost:${port}/?frame=${frame}`); await pg.waitForFunction("window.__done", null, { timeout: 90000 });
  console.log(JSON.stringify(await pg.evaluate("window.__done"))); await pg.screenshot({ path: out }); await b.close(); srv.close();
});
