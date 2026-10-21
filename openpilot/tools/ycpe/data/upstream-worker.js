// the worker behind github.js's upstream(): opendbc master's DBCs from GitHub, checked in or generated (opendbc commits
// only the generators' sources), off the main thread. raw.githubusercontent.com serves the files and jsDelivr lists
// them, both unmetered
import {Generator} from "./opendbc-gen.js";
import {serve} from "./rpc.js";

const REPO = "commaai/opendbc", REF = "master", DIR = "opendbc/dbc/", GEN = `${DIR}generator/`;
const LABEL = `${REPO}@${REF}`;

// a file's text, or null if it doesn't exist
async function get(url) {
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${new URL(url).host} ${r.status}`);
  return r.text();
}
const read = (path) => get(`https://raw.githubusercontent.com/${REPO}/${REF}/${path}`);

class Opendbc {
  constructor(files) {
    this.own = new Set(files.filter((f) => f.startsWith(DIR) && f.endsWith(".dbc") && !f.slice(DIR.length).includes("/")).map((f) => f.slice(DIR.length, -4)));
    this.generator = new Generator(files.filter((f) => f.startsWith(GEN)).map((f) => f.slice(GEN.length)), (f) => read(GEN + f));
    this.names = [...new Set([...this.own, ...this.generator.names.keys()])].sort();
  }

  // one DBC's text
  async text(name) {
    if (this.own.has(name)) return (await read(`${DIR}${name}.dbc`)).replace(/\r\n?/g, "\n");
    if (this.generator.names.has(name)) return this.generator.dbc(name);
    throw new Error(`${name}.dbc is not in opendbc ${LABEL}`);
  }
}

// listed once per worker; offline, a later call tries again
let listed = null;
const opendbc = () => listed ||= get(`https://data.jsdelivr.com/v1/packages/gh/${REPO}@${REF}?structure=flat`).then((text) => {
  if (!text) throw new Error(`no listing of ${LABEL}`);
  return new Opendbc(JSON.parse(text).files.map((f) => f.name.slice(1)));
}).catch((e) => {
  listed = null;
  throw e;
});

serve({
  // every DBC name: {label, names}
  dbcs: async () => ({label: LABEL, names: (await opendbc()).names}),
  dbc: async ({name}) => ({text: await (await opendbc()).text(name)}),
});
