// where a log's schema and DBCs live on GitHub, given initData's git remote and commit. Runs in workers (and node).
import {Checkout, cached} from "./github.js";
import {compile, imports, resolveImport} from "./capnpc.js";
import {Generator} from "./opendbc-gen.js";

// cereal's home over the years: nested openpilot/ (2026 restructure), then in-tree or the cereal submodule (2019-24)
const CEREAL = ["openpilot/cereal", "cereal"];
// the import path cereal is loaded with, holding car.capnp since it moved to opendbc
const IMPORT_PATH = "opendbc_repo/opendbc/car";

// the .capnp files log.capnp needs at a commit: {files: {key: text}, dir}; keys as resolveImport() gives them
export async function schemaFiles(repo, commit) {
  const co = new Checkout(repo, commit);
  const found = await Promise.all(CEREAL.map((d) => co.read(`${d}/log.capnp`)));
  const k = found.findIndex((t) => t !== null);
  if (k < 0) throw new Error(`no log.capnp in ${repo}@${commit.slice(0, 7)}`);
  const dir = CEREAL[k], files = {"log.capnp": found[k]};
  let todo = ["log.capnp"];
  while (todo.length) {
    const keys = new Set();
    for (const from of todo) {
      for (const name of imports(files[from])) {
        const key = resolveImport(from, name);
        if (key !== null && !(key in files)) keys.add(key);
      }
    }
    todo = [...keys];
    const texts = await Promise.all(todo.map((key) => co.read(key.startsWith("/") ? `${IMPORT_PATH}${key}` : `${dir}/${key}`)));
    todo.forEach((key, i) => {
      if (texts[i] === null) throw new Error(`${key} not found in ${repo}@${commit.slice(0, 7)}`);
      files[key] = texts[i];
    });
  }
  return {files, dir};
}

// the compiled schema (schema.json's text) of a commit, compiled once and cached
export async function schemaAt(repo, commit) {
  const t0 = performance.now();
  let fetchMs = 0;
  const text = await cached(commit, `schema/${repo}/${commit}`, async () => {
    const {files} = await schemaFiles(repo, commit);
    fetchMs = performance.now() - t0;
    return compile(files);
  });
  return {text, fetchMs, ms: performance.now() - t0};
}

// ---- opendbc

// a log's opendbc: pinned to its openpilot commit when that resolves, else master. {opendbc, pinned, why}
export async function opendbcFor(repo, commit) {
  let why = "no openpilot commit";
  if (repo && commit) {
    try {
      const co = await opendbcAt(repo, commit), o = co && await Opendbc.open(co);
      if (o) return {opendbc: o, pinned: true};
      why = `no opendbc in ${repo}@${commit.slice(0, 7)}`;
    } catch (e) {
      why = e.message;
    }
  }
  return {opendbc: await Opendbc.open(new Checkout("commaai/opendbc", "master")), pinned: false, why};
}

// opendbc at the commit a log's openpilot used: its submodule, or the tree itself in flattened release branches
export async function opendbcAt(repo, commit) {
  const co = new Checkout(repo, commit);
  const sub = await co.submoduleOf("opendbc_repo/x") ?? await co.submoduleOf("opendbc/x");
  if (sub) return sub.checkout;
  if (await co.read("opendbc_repo/opendbc/__init__.py") !== null) return new Checkout(repo, commit, "opendbc_repo/");
  return null;
}

// the DBCs of an opendbc checkout, checked in (dbc/, or the repo root before 2024) or generated
export class Opendbc {
  constructor(co, files) {
    this.co = co;
    files = files.filter((f) => f.startsWith(co.prefix)).map((f) => f.slice(co.prefix.length));
    this.dir = files.some((f) => f.startsWith("opendbc/dbc/")) ? "opendbc/dbc/" : "";
    const gen = `${this.dir}generator/`;
    this.own = new Set(files.filter((f) => f.startsWith(this.dir) && f.endsWith(".dbc") && !f.slice(this.dir.length).includes("/")).map((f) => f.slice(this.dir.length, -4)));
    this.generator = new Generator(files.filter((f) => f.startsWith(gen)).map((f) => f.slice(gen.length)), (f) => co.read(gen + f));
    this.names = [...new Set([...this.own, ...this.generator.names.keys()])].sort();
  }

  static async open(co) {
    const files = await co.files();
    return files && new Opendbc(co, files);
  }

  // one DBC's text; a checked-in copy (older opendbc committed its generated DBCs) wins
  async text(name) {
    if (this.own.has(name)) return (await this.co.read(`${this.dir}${name}.dbc`)).replace(/\r\n?/g, "\n");
    if (this.generator.names.has(name)) return this.generator.dbc(name);
    throw new Error(`${name}.dbc is not in opendbc ${this.co.label}`);
  }
}
