// GitHub: remotes by name, and opendbc's DBCs through upstream-worker.js
import {client} from "./rpc.js";

// "owner/repo" from any GitHub remote form (https, ssh, scp-like, with or without .git), else null
export function parseRemote(url) {
  const m = /github\.com[/:]([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/.exec(url?.trim() || "");
  return m ? `${m[1]}/${m[2]}` : null;
}

// the main thread's way in: one worker started on first use, for everything fetched from GitHub
let call = null;
export const upstream = (op, args) => (call ||= client(new Worker(new URL("./upstream-worker.js", import.meta.url), {type: "module"})))(op, args);
