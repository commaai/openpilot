// which schema a route's logs are read with: the bundled one plus migrations ("stock"), the log's own exact schema
// fetched from GitHub by the commit that wrote it ("log"), or a local checkout's, live, through the daemon
// ("local:<repo id>"). "auto" picks log for forks and for logs the stock read finds unknown fields in.
import {parseRemote, upstream} from "./github.js";
import {daemon} from "../daemon.js";

const STOCK = "commaai/openpilot";
const controllers = new WeakMap();

// the store's controller, created (and choosing) on first call; route is the API route object
export function logSchema(store, route) {
  if (!controllers.has(store)) controllers.set(store, new LogSchema(store, route));
  return controllers.get(store);
}

// local checkouts the daemon can read schemas from, or null without it
export const localRepos = () => daemon.has("repos") ? daemon.json("/local/repos") : Promise.resolve(null);
export const repoLabel = (r) => `${r.name} · ${r.git.branch}@${r.git.commit}${r.git.dirty ? "*" : ""}`;

class LogSchema {
  constructor(store, route) {
    this.store = store;
    this.route = route;
    this.saved = `ycpe_schema_${route.fullname}`;
    this.mode = localStorage.getItem(this.saved) || "auto";
    this.handlers = new Set();
    this.fetched = new Map();  // key → promise of {key, text, label, ms}
    this.state = {mode: this.mode, active: "stock", label: "stock", status: "", reason: ""};
    this.offs = [store.on("segment", () => this.choose())];
    this.choose();
  }

  on(fn) {
    this.handlers.add(fn);
    fn(this.state);
    return () => this.handlers.delete(fn);
  }

  set(mode) {
    this.mode = mode;
    if (mode === "auto") localStorage.removeItem(this.saved);
    else localStorage.setItem(this.saved, mode);
    this.choose();
  }

  update(state) {
    this.state = {...this.state, ...state, mode: this.mode};
    for (const fn of this.handlers) fn(this.state);
  }

  // the log's git remote and commit: the route's metadata, else initData (the stock read of segment 0 has it)
  git() {
    const valueAt = (p) => this.store.entries.has(p) ? this.store.valueAt(p, Infinity) : undefined;
    const remote = this.route.git_remote || valueAt("/initData/gitRemote");
    const commit = this.route.git_commit || valueAt("/initData/gitCommit");
    return remote && commit ? {repo: parseRemote(remote), commit} : null;
  }

  // what auto mode reads with: [want log, why]
  auto() {
    const git = this.git(), {unknown} = this.store;
    if (!git) return [false, "waiting for initData"];
    if (!git.repo) return [false, "not a GitHub remote"];
    if (git.repo !== STOCK) return [true, `${git.repo} is a fork`];
    if (unknown.events || unknown.services.size) {
      const what = unknown.events ? `${unknown.events} events of unknown services` : `unknown fields in ${[...unknown.services].slice(0, 3).join(", ")}`;
      return [true, `the bundled schema finds ${what}`];
    }
    return [false, "the bundled schema reads this log"];
  }

  choose() {
    const mode = this.mode;
    if (mode.startsWith("local:")) return this.local(mode.slice(6));
    this.unwatch?.();
    this.unwatch = null;
    const [log, reason] = mode === "auto" ? this.auto() : [mode === "log", ""];
    if (!log) return this.use(null, {reason});
    const git = this.git();
    if (!git?.repo) return this.use(null, {status: "error", reason: git ? "not a GitHub remote" : "no git commit in this log"});
    const key = `${git.repo}@${git.commit}`;
    if (!this.fetched.has(key)) {
      this.fetched.set(key, upstream("schema", git).then(({text, ms}) => ({key, text, ms, label: `${git.repo}@${git.commit.slice(0, 7)}`})));
    }
    this.pending(key, this.fetched.get(key), reason);
  }

  // a schema on its way: the stock read goes on meanwhile, and the switch happens only if it's still wanted
  async pending(key, promise, reason) {
    if (this.state.key === key && this.state.status !== "error") return;
    this.update({key, status: "fetching", reason, error: ""});
    try {
      const s = await promise;
      if (this.state.key === key) this.use(s, {reason});
    } catch (e) {
      this.fetched.delete(key);
      if (this.state.key === key) this.use(null, {key, status: "error", error: e.message, reason});
    }
  }

  use(schema, state) {
    const s = this.state;
    if (!schema && !state.status && s.active === "stock" && !s.status && s.reason === state.reason && s.mode === this.mode) return;  // no news
    this.store.setSchema(schema && {key: schema.key, text: schema.text});
    this.update({key: schema?.key ?? state.key ?? null, active: schema ? "log" : "stock", label: schema?.label ?? "stock",
      fork: !!schema && !schema.key.startsWith(`${STOCK}@`), ms: schema?.ms, status: "", error: "", ...state});
  }

  // a local checkout's schema, recompiled whenever its .capnp files change on disk
  async local(id) {
    if (!daemon.has("repos")) return this.use(null, {status: "error", error: "needs the local daemon"});
    if (!this.unwatch) {
      this.version = 0;
      const close = daemon.events("/local/repos/watch", (ev, d) => {
        if (ev === "change" && d.repo === id && d.kind === "capnp" && this.mode === `local:${id}`) {
          this.version++;
          this.local(id);
        }
      });
      this.unwatch = close;
    }
    const key = `local:${id}:${this.version}`;
    this.pending(key, (async () => {
      const [repos, {files}] = await Promise.all([localRepos(), daemon.json(`/local/repos/${id}/capnp`)]);
      const repo = repos.find((r) => r.id === id);
      // the daemon keys files by import path ("car.capnp" for "/car.capnp"); capnpc.js keeps rooted imports apart
      const keyed = Object.fromEntries(Object.entries(files).flatMap(([k, v]) => [[k, v], [`/${k}`, v]]));
      const {text, ms} = await upstream("compile", {files: keyed});
      return {key, text, ms, label: repo ? repoLabel(repo) : id};
    })(), "a local checkout");
  }

  dispose() {
    for (const off of this.offs) off();
    this.unwatch?.();
  }
}
