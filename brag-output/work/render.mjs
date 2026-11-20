// Drive headless Chrome over CDP: seek(t) then screenshot, frame by frame.
// usage: node render.mjs <outDir> <fps> <duration> [t1,t2,... for stills]
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const [outDir, fpsArg, durArg, stillsArg] = process.argv.slice(2);
const fps = Number(fpsArg || 30), dur = Number(durArg || 22.5);
mkdirSync(outDir, { recursive: true });
const port = 9333;
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${resolve(here, "chrome-profile")}`,
  "--hide-scrollbars", "--force-color-profile=srgb", "--window-size=1920,1080", "--allow-file-access-from-files", "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((x) => x.type === "page"); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails)); return r.result?.result?.value; };

await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Page.enable");
await send("Page.navigate", { url: "file://" + resolve(here, "comp.html") });
for (let i = 0; i < 100; i++) { await sleep(100); if (await ev("!!window.ready")) break; }
await ev("window.ready");
await sleep(300);

const times = stillsArg ? stillsArg.split(",").map(Number) : Array.from({ length: Math.round(fps * dur) }, (_, i) => i / fps).slice(Number(process.env.START || 0));
let n = Number(process.env.START || 0);
for (const t of times) {
  await ev(`seek(${t})`);
  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  const name = stillsArg ? `still-${t.toFixed(2)}.png` : `f${String(n).padStart(5, "0")}.png`;
  writeFileSync(resolve(outDir, name), Buffer.from(shot.result.data, "base64"));
  n++;
  if (!stillsArg && n % 60 === 0) console.log(`${n}`);
}
ws.close(); chrome.kill();
console.log("done", n);
