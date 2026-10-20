// read-only GitHub access for workers: files of a repo at a commit, through symlinks and submodules, cached forever
// when keyed by a commit SHA. raw.githubusercontent.com and jsDelivr are unmetered; api.github.com (60 requests an
// hour signed out) is only asked for submodule SHAs, which .gitmodules can't give.

const RAW = "https://raw.githubusercontent.com";
const API = "https://api.github.com";
const JSDELIVR = "https://data.jsdelivr.com/v1/packages/gh";

const SHA = /^[0-9a-f]{40}$/;
const MISSING = "\0missing";  // a cached 404: a path absent at a commit stays absent

// "owner/repo" from any GitHub remote form (https, ssh, scp-like, with or without .git), else null
export function parseRemote(url) {
  const m = /github\.com[/:]([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/.exec(url?.trim() || "");
  return m ? `${m[1]}/${m[2]}` : null;
}

// ---- cache: OPFS files under github/, or whatever setCache() installs (the node tests use a directory)

let cache = {
  async get(key) {
    try {
      let dir = await navigator.storage.getDirectory();
      const parts = ["github", ...key.split("/")];
      for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p);
      return await (await (await dir.getFileHandle(parts.at(-1))).getFile()).text();
    } catch {
      return null;
    }
  },
  async put(key, text) {
    try {
      let dir = await navigator.storage.getDirectory();
      const parts = ["github", ...key.split("/")];
      for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, {create: true});
      const w = await (await dir.getFileHandle(parts.at(-1), {create: true})).createWritable();
      await w.write(text);
      await w.close();
    } catch {}  // no OPFS or over quota: just uncached
  },
};
export const setCache = (c) => { cache = c; };

// immutable things are cached by key when ref is a full SHA; branch names always go to the network
export async function cached(ref, key, fn) {
  if (!SHA.test(ref)) return fn();
  const hit = await cache.get(key);
  if (hit !== null) return hit;
  const v = await fn();
  if (v !== null) await cache.put(key, v);
  return v;
}

export const stats = {requests: 0, api: 0, bytes: 0};

async function get(url, api = false) {
  stats.requests++;
  if (api) stats.api++;
  const r = await fetch(url, api ? {headers: {Accept: "application/vnd.github+json"}} : undefined);
  if (r.status === 404) return null;
  if (!r.ok) {
    const limited = api && (r.status === 403 || r.status === 429);
    throw new Error(limited ? "GitHub API rate limit reached" : `${new URL(url).host} ${r.status}`);
  }
  const text = await r.text();
  stats.bytes += text.length;
  return text;
}

// a submodule URL relative to its superproject resolves like git does: each "../" climbs one level of owner/repo
function submoduleRepo(repo, url) {
  if (!url.startsWith(".")) return parseRemote(url);
  const parts = repo.split("/");
  for (const p of url.replace(/\.git$/, "").split("/")) {
    if (p === "..") parts.pop();
    else if (p !== ".") parts.push(p);
  }
  return parts.length === 2 ? parts.join("/") : null;
}

// a repo at a commit, read like a recursive checkout; prefix roots it at a subdirectory
export class Checkout {
  constructor(repo, ref, prefix = "") {
    this.repo = repo;
    this.ref = ref;
    this.prefix = prefix;
  }

  get label() { return `${this.repo}@${this.ref.slice(0, 7)}`; }

  // a file's text, or null if it doesn't exist; symlinks and submodules are followed
  read(path) { return this.get(this.prefix + path); }

  async get(path, depth = 0) {
    const text = await cached(this.ref, `raw/${this.repo}/${this.ref}/${path}`, async () => (await get(`${RAW}/${this.repo}/${this.ref}/${path}`)) ?? MISSING);
    if (text !== MISSING) {
      // raw serves a symlink as its target path
      if (depth < 4 && text.length < 256 && /^[\w./-]+$/.test(text) && text.includes("/")) return this.get(join(path, text), depth + 1);
      return text;
    }
    const sub = await this.submoduleOf(path);
    return sub ? sub.checkout.get(path.slice(sub.path.length + 1), depth) : null;
  }

  // the submodule containing path: {path, checkout}, or null
  async submoduleOf(path) {
    if (path === ".gitmodules") return null;
    this.modules ??= this.get(".gitmodules").then(parseGitmodules);
    const mod = (await this.modules).find((m) => path.startsWith(`${m.path}/`));
    if (!mod) return null;
    const repo = submoduleRepo(this.repo, mod.url);
    const sha = repo && await this.submoduleSha(mod.path);
    return sha ? {path: mod.path, checkout: new Checkout(repo, sha)} : null;
  }

  async submoduleSha(path) {
    return cached(this.ref, `sub/${this.repo}/${this.ref}/${path}`, async () => {
      const r = await get(`${API}/repos/${this.repo}/contents/${path}?ref=${this.ref}`, true);
      const j = r && JSON.parse(r);
      return j?.type === "submodule" ? j.sha : null;
    });
  }

  // every file path in the repo (not its submodules), from jsDelivr
  async files() {
    const text = await cached(this.ref, `ls/${this.repo}/${this.ref}`, () => get(`${JSDELIVR}/${this.repo}@${this.ref}?structure=flat`));
    if (!text) return null;
    return JSON.parse(text).files.map((f) => f.name.slice(1));
  }
}

function parseGitmodules(text) {
  const out = [];
  for (const block of (text || "").split(/^\[submodule/m).slice(1)) {
    const path = /^\s*path\s*=\s*(.+?)\s*$/m.exec(block)?.[1], url = /^\s*url\s*=\s*(.+?)\s*$/m.exec(block)?.[1];
    if (path && url) out.push({path: path.replace(/\/$/, ""), url});
  }
  return out;
}

// a link target resolved from the link's own directory
function join(path, target) {
  const parts = path.split("/").slice(0, -1);
  for (const p of target.split("/")) {
    if (p === "..") parts.pop();
    else if (p && p !== ".") parts.push(p);
  }
  return parts.join("/");
}
