// the worker behind github.js's upstream(): GitHub fetching, schema compiling and DBC generating, off the main thread
import {schemaAt, opendbcFor} from "./upstream.js";
import {compile} from "./capnpc.js";
import {serve} from "./rpc.js";

const opendbcs = new Map();  // "repo@commit" → opendbcFor()

const OPS = {
  // a commit's compiled schema: {text, fetchMs, ms}
  schema: ({repo, commit}) => schemaAt(repo, commit),
  // files ({key: text}, keys as capnpc.resolveImport gives them) compiled: {text, ms}
  compile({files}) {
    const t0 = performance.now(), text = compile(files);
    return {text, ms: performance.now() - t0};
  },
  // the DBCs of a log's opendbc: {label, names, pinned, why}
  async dbcs({repo, commit}) {
    const key = `${repo}@${commit}`;
    if (!opendbcs.has(key)) opendbcs.set(key, opendbcFor(repo, commit));
    const {opendbc, pinned, why} = await opendbcs.get(key).catch((e) => {
      opendbcs.delete(key);  // offline: let a later call retry
      throw e;
    });
    return {label: opendbc.co.label, names: opendbc.names, pinned, why};
  },
  async dbc({repo, commit, name}) {
    await OPS.dbcs({repo, commit});
    return {text: await (await opendbcs.get(`${repo}@${commit}`)).opendbc.text(name)};
  },
};

serve(OPS);
