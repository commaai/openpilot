// the route info row for the schema the logs are read with (data/schemas.js): what's in use, why, and the choice
import {h, css} from "/dom.js";
import {logSchema, localRepos, repoLabel} from "/data/schemas.js";
import {needsDaemon} from "/daemon.js";

await css(new URL("./schema.css", import.meta.url));

const MODES = [["auto", "auto"], ["stock", "stock (bundled, migrated)"], ["log", "log (its own commit)"]];

const took = (ms) => ms < 10 ? "cached" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

// el, filled in once the route's logs start loading (the info pane doesn't load them itself); returns a dispose()
export function schemaRow(ctx) {
  const el = h("span", {class: "schema-row muted"}, "chosen when the logs load (open Plot or CAN)");
  let off = null;
  const unsub = ctx.onStore((store) => {
    const ls = logSchema(store, ctx.route);
    const select = h("select", {class: "schema-select", title: "The schema this route's logs are read with", onchange: () => ls.set(select.value)},
      MODES.map(([v, label]) => h("option", {value: v}, label)));
    const what = h("span", {class: "mono"}), note = h("span", {class: "small"});
    localRepos().then((repos) => {
      if (!repos) return select.append(h("option", {disabled: true, title: needsDaemon("A local checkout schema")}, "local checkout (needs the daemon)"));
      select.append(...repos.filter((r) => r.cereal).map((r) => h("option", {value: `local:${r.id}`}, `local: ${repoLabel(r)}`)));
      select.value = ls.mode;
    }, () => {});
    off = ls.on((s) => {
      select.value = s.mode;
      what.textContent = s.label;
      note.className = `small ${s.status === "error" ? "error" : "muted"}`;
      note.textContent = s.status === "fetching" ? `fetching ${s.key}…`
        : s.status === "error" ? `${s.error || s.reason}; reading with the bundled schema`
        : [s.mode === "auto" && s.reason, s.ms !== undefined && took(s.ms)].filter(Boolean).join(" · ");
    });
    el.className = "schema-row";
    el.replaceChildren(what, select, note);
  });
  return {el, dispose: () => { off?.(); unsub?.(); }};
}
