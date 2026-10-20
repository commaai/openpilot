// the "Re-run openpilot" pane: replay processes from a local checkout over a segment range (a daemon job), follow
// its progress, load the replayed log as a store overlay under /replay/<id>, and list the signals that diverged.
import {h, css, fmt} from "/dom.js";
import {daemon, needsDaemon} from "/daemon.js";
import {onJobs, allJobs} from "/local.js";
import {compare as plotCompare} from "/panes/plot/view.js";

const SEGMENT = 60;

export function create(ctx, state) {
  css(new URL("./replay.css", import.meta.url));
  const el = h("div", {class: "replay-pane"});
  const pane = {name: "replay", el, render() {}, resize() {}, save: () => ({procs: [...picked], repo: repoSel.value || state.repo}), dispose};
  const offs = [];
  function dispose() { for (const off of offs) off(); }

  if (!daemon.has("replay")) {
    el.append(h("div", {class: "replay-note muted"}, needsDaemon("Re-running openpilot")));
    return pane;
  }

  const route = ctx.route.fullname;
  const segCount = (ctx.route.maxqlog ?? 0) + 1;
  const picked = new Set(state.procs || daemon.caps.modules.replay.default);
  const repoSel = h("select", {class: "replay-repo", onchange: () => loadProcs()});
  const commit = h("span", {class: "muted mono"});
  const procs = h("div", {class: "replay-procs"});
  const lacking = h("div", {class: "replay-missing"});
  const first = h("input", {type: "number", min: 0, max: segCount - 1, class: "replay-seg"});
  const last = h("input", {type: "number", min: 0, max: segCount - 1, class: "replay-seg"});
  const start = h("button", {class: "chip replay-start", onclick: () => run()}, "Re-run");
  const status = h("div", {class: "replay-status"});
  const bar = h("i");
  const progress = h("div", {class: "replay-bar", hidden: true}, bar);
  const errors = h("pre", {class: "replay-error", hidden: true});
  const diff = h("div", {class: "replay-diff"});
  const history = h("div", {class: "replay-history"});

  const visible = () => {
    const [t0, t1] = ctx.clock.range;
    first.value = Math.max(0, Math.min(segCount - 1, Math.floor(t0 / SEGMENT)));
    last.value = Math.max(+first.value, Math.min(segCount - 1, Math.floor(Math.max(t0, t1 - 1e-3) / SEGMENT)));
  };
  visible();

  el.append(
    h("div", {class: "replay-row"}, h("label", {}, "checkout "), repoSel, commit),
    procs, lacking,
    h("div", {class: "replay-row"}, "segments ", first, " to ", last,
      h("button", {class: "link", title: "The segments of the visible x-range", onclick: visible}, "visible range"), h("span", {class: "grow"}), start),
    h("div", {class: "muted"}, "Processes start on the segment before the range, when there is one, so engagement, calibration and learners are warm when it begins; only the range is written and compared."),
    progress, status, errors,
    h("h3", {}, "Diverged signals"), diff,
    h("h3", {}, "Replays of this route"), history);

  let current = null;  // the replay this pane follows: {id, job}
  let known = {};      // process name → missing parts in the selected checkout

  async function loadRepos() {
    const repos = daemon.has("repos") ? await daemon.json("/local/repos").catch(() => []) : [];
    repoSel.replaceChildren(...repos.map((r) => h("option", {value: r.path}, `${r.name}${r.branch ? ` (${r.branch})` : ""}`)));
    if (state.repo && repos.some((r) => r.path === state.repo)) repoSel.value = state.repo;
    loadProcs();
  }

  async function loadProcs() {
    const q = repoSel.value ? `?repo=${encodeURIComponent(repoSel.value)}` : "";
    let info;
    try {
      info = await daemon.json(`/local/replay/procs${q}`);
    } catch (e) {
      lacking.textContent = e.message;
      return;
    }
    if (!repoSel.options.length) repoSel.append(h("option", {value: info.repo}, info.repo.split("/").at(-1)));  // no repos module: the daemon's checkout
    commit.textContent = info.commit ? info.commit.slice(0, 8) : "";
    known = Object.fromEntries(info.procs.map((p) => [p.name, p.missing]));
    procs.replaceChildren(...info.procs.map(({name, missing, videos = []}) => {
      const box = h("input", {type: "checkbox", checked: picked.has(name), onchange: (e) => { e.target.checked ? picked.add(name) : picked.delete(name); check(); }});
      const title = missing.length ? `missing: ${missing.join(", ")}` : videos.length ? `fed frames from ${videos.join(" and ")}, downloaded per segment` : "";
      return h("label", {class: `replay-proc${missing.length ? " lacking" : ""}`, title}, box, name);
    }));
    lacking.dataset.all = info.missing.join(", ");
    check();
  }

  // says exactly what the checkout lacks for the chosen processes, before anything runs
  function check() {
    const gaps = [];
    if (lacking.dataset.all) gaps.push(`all processes: ${lacking.dataset.all}`);
    for (const p of picked) if (known[p]?.length) gaps.push(`${p}: ${known[p].join(", ")}`);
    lacking.replaceChildren(...(gaps.length ? [h("b", {}, "Not built in this checkout (run scons there): "), gaps.join("; ")] : []));
    start.disabled = !picked.size || gaps.length > 0;
  }

  async function run() {
    errors.hidden = true;
    diff.replaceChildren();
    const body = {route, segs: [+first.value, +last.value], procs: [...picked], repo: repoSel.value || undefined};
    try {
      const m = await daemon.json("/local/replay", {method: "POST", body});
      follow(m.id, m.job);
    } catch (e) {
      status.textContent = e.message;
    }
  }

  function follow(id, job) {
    current = {id, job};
    start.textContent = "Cancel";
    start.onclick = () => daemon.fetch(`/local/jobs/${job}/cancel`, {method: "POST"});
    start.disabled = false;
    const j = allJobs().find((x) => x.id === job);
    if (j) update(j);
  }

  function update(job) {
    if (!current || job.id !== current.job) return;
    const running = job.state === "running";
    progress.hidden = !running;
    bar.style.width = `${job.total ? (100 * job.done / job.total).toFixed(1) : 0}%`;
    status.textContent = running ? job.text || "starting" : job.state === "done" ? `done in ${job.result?.seconds?.toFixed(0) ?? "?"} s` : job.state;
    attach(current.id, job.meta.written || []);
    if (running) return;
    start.textContent = "Re-run";
    start.onclick = () => run();
    check();
    if (job.state === "error") {
      errors.hidden = false;
      errors.textContent = job.error;
    }
    if (job.state === "done") showDiff(job.result);
    current = null;
    loadHistory();
  }

  // the replayed segments as a store overlay, growing as they're written
  const attached = new Map();  // id → segments attached
  function attach(id, segs) {
    const store = ctx.store;  // replays are for plotting: this starts the route's logs too
    if (!store?.attach || !segs.length || attached.get(id) === segs.length) return;
    attached.set(id, segs.length);
    const urls = [];
    for (const n of segs) urls[n] = daemon.url(`/local/replay/${id}/${n}/rlog.zst`);
    store.attach(`/replay/${id}`, urls, {name: `replay/${id}`});
  }

  function showDiff(m) {
    const d = m?.diff;
    if (!d) return diff.replaceChildren(h("div", {class: "muted"}, "no summary"));
    const svc = Object.entries(d.services).map(([s, c]) => `${s} ${c.replayed}/${c.original}`).join(" · ");
    const rows = d.signals.map((s) => h("tr", {class: "replay-signal", title: "Compare in the plot and seek to the first difference", onclick: () => compare(m, s)},
      h("td", {class: "mono path"}, s.path),
      h("td", {}, `${(100 * s.differ).toFixed(s.differ < 0.01 ? 2 : 0)}%`),
      h("td", {}, num(s.meanAbs)), h("td", {}, num(s.maxAbs)),
      h("td", {}, s.first == null ? "" : fmt.clock(s.first))));
    const copy = h("button", {class: "link", title: "A markdown table of the diverged signals, for a PR", onclick: async () => {
      try {
        await navigator.clipboard.writeText(markdown(m));
        copy.textContent = "copied";
      } catch {
        copy.textContent = "copy failed";
      }
      setTimeout(() => { copy.textContent = "copy as markdown"; }, 1500);
    }}, "copy as markdown");
    diff.replaceChildren(
      h("div", {class: "replay-row"}, h("span", {class: "muted grow", title: svc}, `${d.diverged} of ${d.compared} signals differ · ${svc}`), copy),
      rows.length ? h("table", {}, h("thead", {}, h("tr", {}, ["signal", "differs", "mean |Δ|", "max |Δ|", "first"].map((c) => h("th", {}, c)))),
        h("tbody", {}, rows)) : h("div", {}, "identical to the original log"));
  }

  // the original and the replayed signal as a pair in a plot, at the first difference
  function compare(m, s) {
    attach(m.id, m.written);
    if (!plotCompare(s.path, `/replay/${m.id}`, {t: s.first ?? undefined})) status.textContent = "open a plot pane to compare signals";
  }

  async function loadHistory() {
    const list = await daemon.json(`/local/replay?route=${encodeURIComponent(route)}`).catch(() => []);
    history.replaceChildren(...(list.length ? list.map((m) => h("div", {class: `replay-past ${m.state}`},
      h("span", {class: "grow"}, `${m.procs.join(", ")} · segments ${m.segs[0]}–${m.segs.at(-1)} · ${m.commit?.slice(0, 8) || "?"}`),
      h("span", {class: "muted"}, m.state === "done" ? `${m.diff?.diverged ?? 0} diverged` : m.state),
      m.written.length > 0 && h("button", {class: "link", onclick: () => { attach(m.id, m.written); showDiff(m); }}, "load"))) : [h("div", {class: "muted"}, "none yet")]));
  }

  offs.push(onJobs(update));
  loadRepos();
  loadHistory();
  return pane;
}

// the diff summary as markdown for a PR: what ran where, then the diverged signals
export function markdown(m) {
  const d = m.diff, segs = m.segs[0] === m.segs.at(-1) ? `segment ${m.segs[0]}` : `segments ${m.segs[0]}–${m.segs.at(-1)}`;
  const lines = [
    `Replay of ${m.procs.join(", ")} at \`${m.commit?.slice(0, 8) || "?"}\` on \`${m.route}\` ${segs}` +
      `${m.warm != null ? ` (warmed up on segment ${m.warm})` : ""}: ${d.diverged} of ${d.compared} signals differ.`,
    "",
  ];
  if (d.signals.length) {
    lines.push("| signal | differs | mean \\|Δ\\| | max \\|Δ\\| | first |", "| --- | ---: | ---: | ---: | ---: |");
    for (const s of d.signals) {
      lines.push(`| \`${s.path}\` | ${(100 * s.differ).toFixed(s.differ < 0.01 ? 2 : 0)}% | ${num(s.meanAbs)} | ${num(s.maxAbs)} | ${s.first == null ? "" : fmt.clock(s.first)} |`);
    }
    if (d.diverged > d.signals.length) lines.push("", `${d.diverged - d.signals.length} more not shown.`);
  }
  return lines.join("\n") + "\n";
}

const num = (x) => (x === 0 ? "0" : Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toPrecision(3));
