// what node scripts share: the repo, its python, the log to read, and the app's decoding (codec.wasm, bundled schema)
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {initCodec} from "../../codec/codec.js";

export const REPO = new URL("../../../../../", import.meta.url).pathname;
// this checkout's venv, else the main checkout's (worktrees share it)
const common = () => path.dirname(execFileSync("git", ["-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"]).toString().trim());
export const PYTHON = process.env.PYTHON || path.join(fs.existsSync(path.join(REPO, ".venv")) ? REPO : common(), ".venv/bin/python");
export const schema = JSON.parse(fs.readFileSync(new URL("../schema.json", import.meta.url)));

export function logArg(k = 2) {
  const log = process.argv[k] || process.env.YCPE_RLOG;
  if (!log) throw new Error("no log: pass one, or set YCPE_RLOG to an rlog (compressed or not)");
  return log;
}

// initCodec fetches codec.wasm, so file: URLs read from disk
const netFetch = globalThis.fetch;
globalThis.fetch = async (u, o) => String(u).startsWith("file:") ? new Response(fs.readFileSync(new URL(u)), {headers: {"content-type": "application/wasm"}}) : netFetch(u, o);
await initCodec();
