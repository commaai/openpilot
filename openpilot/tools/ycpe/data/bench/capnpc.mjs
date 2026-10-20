// parity of capnpc.js with pycapnp: for commits across cereal's history and forks, fetches the schema files the way
// the app does (upstream.js), compiles them in JS and with gen_schema.py (capnpc_ref.py), and diffs the text.
// usage: PYTHONPATH=<repo root> node capnpc.mjs [--fresh] [repo@sha ...]
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {setCache, stats} from "../github.js";
import {schemaFiles} from "../upstream.js";
import {compile} from "../capnpc.js";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const PYTHON = process.env.PYTHON || "/home/batman/openpilot/.venv/bin/python";

const COMMITS = [
  ["commaai/openpilot", "83b249571ba7679d1d4bd2611219fe97dcd1a21a", "2019-01 in-tree cereal subtree"],
  ["commaai/openpilot", "3bae5ab6e849f714b7d1fd8aa283e03a36c352b9", "2020-03 cereal submodule"],
  ["commaai/openpilot", "8f9f31778d2683d31737ad07fd271213056728ca", "2020-10 cereal submodule"],
  ["commaai/openpilot", "2d6d92402a152e5f53c5f41be7b560cf4993a77c", "2021-04 cereal submodule"],
  ["commaai/openpilot", "88ebf835bbd07a12e85bf79dd0193bf2d3b23834", "2021-12 cereal submodule"],
  ["commaai/openpilot", "f50800b7684862b09cba23415b5496ad9b779552", "2022-09 cereal submodule"],
  ["commaai/openpilot", "9897bc83c44c77d9ba03f909cf83bd518d239e5b", "2023-03 cereal submodule"],
  ["commaai/openpilot", "b952317e9bd5db2af2a99749ddd223b65e809880", "2023-08 cereal submodule"],
  ["commaai/openpilot", "3d19b4fe3a6a1078292e4f4eab851a697581969e", "2024-04 cereal submodule"],
  ["commaai/openpilot", "24c18d2c4b05968bc3166d8af53851dd966c4f54", "2024-08 in-tree cereal"],
  ["commaai/openpilot", "73ff9dc944f05941729fa979db32ddacd573e5df", "2025-05 in-tree, car.capnp symlink"],
  ["commaai/openpilot", "7099bd18e3ca5936cdd3c8de8a8c679cc7edb053", "2026-02 in-tree"],
  ["commaai/openpilot", "9844075bb27fee142a8e123a9e03bdf1afd1e008", "2026-05 in-tree"],
  ["commaai/openpilot", "4f3626dc9b4d929c9ee0c8feceb9c2b80e7a950c", "2026-10 restructured, /car.capnp in opendbc"],
  ["commaai/openpilot", "2ab952d5eb16020116d3354b83f30303db5a99b4", "master"],
  ["commaai/openpilot", "551d088497144425feb14299c6d725b63f1b942e", "release3 (flattened)"],
  ["commaai/openpilot", "70e157462304e5ce7d03ffbec6cb7f45bf347bb7", "release-mici (flattened)"],
  ["sunnypilot/sunnypilot", "922513f3128b00611d695ce465989a9bc8af9d59", "sunnypilot master"],
  ["sunnypilot/sunnypilot", "5ba3d8ea92603fff5edddaf25ee3e679f791f277", "sunnypilot release-c3"],
  ["sunnypilot/sunnypilot", "35c37ca463c5c674b068878f58cab0e284226b3a", "sunnypilot dev"],
  ["FrogAI/FrogPilot", "1e23dec6352cef5a36a87be0af7d7a082b7c48a4", "FrogPilot"],
  ["dragonpilot-community/dragonpilot", "741c1d1a706ebb2872efd58ac59cfdfa8d3f20ee", "dragonpilot 0.10.2"],
];

// the app caches in OPFS; here a directory stands in
const CACHE = path.join(os.tmpdir(), "ycpe-github");
if (process.argv.includes("--fresh")) fs.rmSync(CACHE, {recursive: true, force: true});
setCache({
  async get(key) { try { return fs.readFileSync(path.join(CACHE, key), "utf8"); } catch { return null; } },
  async put(key, text) { fs.mkdirSync(path.dirname(path.join(CACHE, key)), {recursive: true}); fs.writeFileSync(path.join(CACHE, key), text); },
});

// annotation-only files capnp needs on disk (capnpc.js skips them)
const CXX = fs.readFileSync(path.join(execFileSync(PYTHON, ["-c", "import capnp, os; print(os.path.dirname(capnp.__file__))"]).toString().trim(), "c++.capnp"), "utf8");
const JAVA = "@0xc5f1af96651f70ea;\nannotation package @0x9ee4c8f803b3b596 (file) :Text;\nannotation outerClassname @0x9b066bb4881f7cd3 (file) :Text;\n";

// --fuzz N: random structs dense with unions in unions, groups in unions and mixed sizes, which stress the layout's
// hole filling and union expansion harder than cereal does
function fuzz(seed) {
  let s = seed;
  const rnd = (n) => (s = (s * 1103515245 + 12345) % 2147483648) % n;
  const TYPES = ["Bool", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Float32", "Int64", "UInt64", "Float64", "Text", "Data", "Void", "List(Int32)", "E", "S"];
  const DEFAULTS = {Bool: "true", Int8: "-5", UInt16: "7", Int32: "-100000", UInt64: "12345678901234567890", Float32: "0.05", Float64: "-1.5e300", E: "b"};
  let ordinal = 0, name = 0;
  const fields = [];
  const member = (depth) => {
    const k = depth < 3 ? rnd(10) : 0;
    if (k < 6) {
      const t = TYPES[rnd(TYPES.length)], o = ordinal++;
      fields.push(o);
      const def = DEFAULTS[t] && rnd(3) === 0 ? ` = ${DEFAULTS[t]}` : "";
      return `f${name++} @${o} :${t}${def};`;
    }
    if (k < 8) return `u${name++} :union { ${Array.from({length: 2 + rnd(3)}, () => member(depth + 1)).join(" ")} }`;
    return `g${name++} :group { ${Array.from({length: 1 + rnd(3)}, () => member(depth + 1)).join(" ")} ${rnd(3) === 0 ? `union { ${member(depth + 1)} ${member(depth + 1)} }` : ""} }`;
  };
  const body = Array.from({length: 3 + rnd(10)}, () => member(0)).join("\n  ");
  const top = rnd(2) ? `union { ${member(1)} ${member(1)} ${member(1)} }` : "";
  // capnp wants ordinals 0..n-1 but not in declaration order: shuffle them
  const perm = fields.map((_, i) => i);
  for (let i = perm.length - 1; i > 0; i--) { const j = rnd(i + 1); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const event = `struct Event {\n  ${body}\n  ${top}\n}\n`.replace(/@(\d+) :/g, (_, o) => `@${perm[+o]} :`);
  return {"log.capnp": `@0x${(0x8000000000000000n + BigInt(seed) * 7919n).toString(16)};\nenum E { a @0; b @1; c @2; }\nstruct S { x @0 :Int32; }\n${event}`};
}

const args = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const nfuzz = process.argv.includes("--fuzz") ? +args.shift() : 0;
const list = nfuzz ? Array.from({length: nfuzz}, (_, i) => ["fuzz", i + 1, `fuzz seed ${i + 1}`]) : args.length ? args.map((a) => [...a.split("@"), a]) : COMMITS;
let failed = 0, skipped = 0;
for (const [repo, sha, label] of list) {
  const before = {...stats};
  let t0 = performance.now(), files;
  try {
    ({files} = repo === "fuzz" ? {files: fuzz(sha)} : await schemaFiles(repo, sha));
  } catch (e) {
    console.log(`FAIL ${label}: ${e.message}`);
    failed++;
    continue;
  }
  const fetchMs = performance.now() - t0;
  t0 = performance.now();
  const js = compile(files);
  const compileMs = performance.now() - t0;

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "capnpc-"));
  for (const [key, text] of Object.entries(files)) {
    const p = path.join(dir, key.startsWith("/") ? "imp" : "cereal", key);
    fs.mkdirSync(path.dirname(p), {recursive: true});
    fs.writeFileSync(p, text);
  }
  for (const sub of ["imp/include", "cereal/include"]) {
    fs.mkdirSync(path.join(dir, sub), {recursive: true});
    fs.writeFileSync(path.join(dir, sub, "c++.capnp"), CXX);
    fs.writeFileSync(path.join(dir, sub, "java.capnp"), JAVA);
  }
  t0 = performance.now();
  let ref;
  try {
    ref = execFileSync(PYTHON, [path.join(HERE, "capnpc_ref.py"), dir], {stdio: ["ignore", "pipe", "pipe"], maxBuffer: 1 << 26}).toString();
  } catch (e) {
    const err = e.stderr.toString().split("\n"), msg = err.find((l) => /failed|Error/.test(l)) ?? err.at(-2);
    // capnp refuses schemas its pre-0.6 layout got wrong (issue 344); no log can have one, so they aren't cases
    if (repo === "fuzz" && msg.includes("issues/344")) {
      skipped++;
      continue;
    }
    console.log(`FAIL ${label}: pycapnp: ${msg}`);
    failed++;
    continue;
  }
  const refMs = performance.now() - t0;
  fs.rmSync(dir, {recursive: true});

  const s = JSON.parse(js), same = js === ref;
  let at = 0;
  while (!same && js[at] === ref[at]) at++;
  if (!same) failed++;
  console.log(`${same ? "ok  " : "DIFF"} ${label.padEnd(42)} ${repo}@${String(sha).slice(0, 7)} ${Object.keys(files).length} files, ${s.structs.length} structs, ${s.enums.length} enums, ` +
    `${(js.length / 1024).toFixed(0)} KB | fetch ${fetchMs.toFixed(0)} ms (${stats.requests - before.requests} req, ${stats.api - before.api} api) | ` +
    `compile ${compileMs.toFixed(1)} ms, pycapnp ${refMs.toFixed(0)} ms` + (same ? "" : `\n     js:  …${js.slice(Math.max(0, at - 80), at + 80)}\n     ref: …${ref.slice(Math.max(0, at - 80), at + 80)}`));
}
const ran = list.length - skipped;
console.log((failed ? `${failed} of ${ran} failed` : `all ${ran} identical`) + (skipped ? ` (${skipped} schemas capnp rejects skipped)` : ""));
process.exit(failed ? 1 : 0);
