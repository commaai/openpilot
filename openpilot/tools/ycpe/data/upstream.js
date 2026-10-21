// where a log's DBCs live on GitHub, given initData's git remote and commit. Runs in workers (and node).
import {Checkout} from "./github.js";
import {Generator} from "./opendbc-gen.js";

// a log's opendbc: pinned to its openpilot commit when that resolves, else master. {opendbc, pinned, why}
export async function opendbcFor(repo, commit) {
  let why = "no openpilot commit";
  if (repo && commit) {
    try {
      const co = await opendbcAt(repo, commit), o = co && (co.prefix ? new Flat(co, await master()) : await Opendbc.open(co));
      if (o) return {opendbc: o, pinned: true};
      why = `no opendbc in ${repo}@${commit.slice(0, 7)}`;
    } catch (e) {
      why = e.message;
    }
  }
  return {opendbc: await master(), pinned: false, why};
}

let masterOpendbc = null;
const master = () => masterOpendbc ||= Opendbc.open(new Checkout("commaai/opendbc", "master")).catch((e) => {
  masterOpendbc = null;
  throw e;
});

// opendbc at the commit a log's openpilot used: its submodule, or the tree itself in flattened release branches
export async function opendbcAt(repo, commit) {
  const co = new Checkout(repo, commit);
  // release branches (what most devices run) have it in-tree: checked first, so they never ask for a .gitmodules
  if (await co.raw("opendbc_repo/opendbc/__init__.py") !== null) return new Checkout(repo, commit, "opendbc_repo/");
  const sub = await co.submoduleOf("opendbc_repo/x") ?? await co.submoduleOf("opendbc/x");
  return sub?.checkout ?? null;
}

// opendbc in a flattened release: its DBCs, generated ones too, are built and committed, but listing a repo that big
// takes the rate-limited API (jsDelivr won't), so the names are opendbc master's and a DBC the release lacks is master's
class Flat {
  constructor(co, master) {
    this.co = co;
    this.master = master;
    this.names = master.names;
  }

  async text(name) {
    const text = await this.co.read(`opendbc/dbc/${name}.dbc`);
    return text === null ? this.master.text(name) : text.replace(/\r\n?/g, "\n");
  }
}

// the DBCs of an opendbc checkout, checked in (dbc/, or the repo root before 2024) or generated
export class Opendbc {
  constructor(co, files) {
    this.co = co;
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
