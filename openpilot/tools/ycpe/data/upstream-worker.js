// the worker behind github.js's upstream(): GitHub fetching and DBC generating, off the main thread
import {opendbcFor} from "./upstream.js";
import {serve} from "./rpc.js";

const opendbcs = new Map();  // "repo@commit" → opendbcFor()

const OPS = {
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
