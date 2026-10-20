// boot and crash logs: the signed file list, newest first. Crash logs open in a text dialog, boot logs (capnp) in a
// viewer of their sections.
import {api} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";

const PAGE = 50;

// "2026-08-16--02-07-04_e8fd3854_openpilot_system_loggerd_loggerd" → when, commit, executable
function crashName(file) {
  const m = file.match(/^(\d{4}-\d\d-\d\d)--(\d\d)-(\d\d)-(\d\d)_([0-9a-f]+)_(.*)$/);
  return m ? [`${m[1]} ${m[2]}:${m[3]}:${m[4]}`, m[5], m[6].replaceAll("_", "/")] : [file, "", ""];
}

// closing returns the focus to what opened it
function logDialog(name, head, body) {
  const opener = document.activeElement;
  const dialog = h("dialog", {class: "log-dialog", onclose: () => { dialog.remove(); opener?.focus(); }},
    h("div", {class: "log-head"},
      h("span", {class: "mono"}, name),
      head,
      h("button", {class: "icon-btn", "aria-label": "Close", onclick: () => dialog.close()}, icon("close", 20))),
    body);
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  document.body.append(dialog);
  dialog.showModal();
  return dialog;
}

function openCrash(url, name) {
  const pre = h("pre", {class: "log-text"}, "Loading…");
  fetch(url).then((r) => r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then((text) => { pre.textContent = text; }, (err) => { pre.textContent = String(err); });
  logDialog(name, h("a", {class: "icon-btn", href: url, download: name, "aria-label": "Download"}, icon("download", 20)), pre);
}

// a boot log read by a log worker (data/boot.js), so a big one never holds up a frame; aborting stops the worker
function readBoot(url, signal) {
  const w = new Worker(new URL("../data/worker.js", import.meta.url), {type: "module"});
  return new Promise((resolve, reject) => {
    w.onmessage = ({data}) => data.error ? reject(new Error(data.error)) : resolve(data.boot);
    w.onerror = () => reject(new Error("couldn't start the log worker"));
    signal.addEventListener("abort", () => reject(new Error("closed")));
    w.postMessage({id: 0, op: "boot", url});
  }).finally(() => w.terminate());
}

// "key: value" lines, or a block under the key for values over several lines
const pairs = (entries) => entries.map(([k, v]) => v.trim().includes("\n") ? `${k}:\n${v.trimEnd()}\n` : `${k}: ${v.trim()}`).join("\n");

// the viewer's sections as [title, text, open]: what the boot ran first, then what the device was
function bootSections(b) {
  const init = b.init;
  return [
    ["launch log", b.launchLog, true],
    ...b.commands.map(([cmd, out]) => [cmd, out, true]),
    ...(b.pstore.length ? b.pstore.map(([file, out]) => [`pstore/${file}`, out, true]) : [["pstore", "", false]]),
    ...[["last kmsg", b.lastKmsg, true], ["last pmsg", b.lastPmsg, true]].filter(([, text]) => text),
    ...(init ? [
      ["device", pairs([["kernel", init.kernelVersion], ["os", init.osVersion], ...init.commands]), false],
      ["kernel args", init.kernelArgs.join("\n"), false],
      ["params", pairs(init.params), false],
    ] : []),
  ];
}

// bootlog runs before the clock is synced, so a device that lost its time stamps 1970 or a stale date
const CLOCK_SET = Date.UTC(2020, 0, 1);

// "0.11.2 · master · 7e84a616 (dirty) · mici"
function bootSummary(b) {
  const i = b.init, when = new Date(b.wallTimeNanos / 1e6);
  const what = i && [i.version, i.gitBranch, i.gitCommit && `${i.gitCommit.slice(0, 8)}${i.dirty ? " (dirty)" : ""}`, i.deviceType].filter(Boolean).join(" · ");
  return h("div", {class: "boot-summary"},
    when >= CLOCK_SET
      ? h("span", {}, "booted ", h("b", {}, when.toLocaleString()), h("span", {class: "muted"}, ` (${fmt.ago(when / 1000)})`))
      : h("span", {class: "muted"}, "boot time unknown (clock not set)"),
    what && h("span", {class: "mono muted"}, what));
}

// lines go in blocks of CHUNK that skip layout while off screen (content-visibility), so a long log opens fast
const CHUNK = 100;

// line with every q in it marked
function marked(line, q) {
  const lower = line.toLowerCase(), out = [];
  let at = 0;
  for (let i = lower.indexOf(q); i >= 0; i = lower.indexOf(q, at)) {
    out.push(line.slice(at, i), h("mark", {}, line.slice(i, i + q.length)));
    at = i + q.length;
  }
  out.push(line.slice(at));
  return out;
}

// one section's shown lines, for filter q (lowercase): all of them, or the ones with q in them, each block handed to
// mark(block, lines) to have q marked later
function render(s, q, mark) {
  const shown = q ? s.lines.filter((l) => l.toLowerCase().includes(q)) : s.lines;
  const blocks = [];
  for (let k = 0; k < shown.length; k += CHUNK) {
    const lines = shown.slice(k, k + CHUNK), block = h("div", {}, lines.join("\n"));
    if (q) mark(block, lines);
    blocks.push(block);
  }
  s.pre.replaceChildren(...(s.lines.length ? blocks : ["(empty)"]));
  s.pre.classList.toggle("muted", !s.lines.length);
  s.count.textContent = !s.lines.length ? "" : q ? `${shown.length} of ${s.lines.length} lines` : `${s.lines.length} lines`;
  return shown;
}

function openBoot(url, name) {
  const ctl = new AbortController();
  const filter = h("input", {type: "search", class: "field boot-filter", placeholder: "Filter lines  (ctrl+f)", spellcheck: false});
  const matches = h("span", {class: "muted small nowrap"});
  const copyBtn = (text) => {
    const btn = h("button", {class: "icon-btn", title: "Copy what's shown", "aria-label": "Copy", onclick: async (e) => {
      e.preventDefault();  // in a section's summary, copying doesn't toggle it
      let ok = true;
      try {
        await navigator.clipboard.writeText(text());
      } catch {
        ok = false;
      }
      btn.title = ok ? "Copied" : "Couldn't copy: the clipboard was denied";
      btn.replaceChildren(icon(ok ? "check" : "warning", 18));
      setTimeout(() => {
        btn.title = "Copy what's shown";
        btn.replaceChildren(icon("copy", 18));
      }, 1500);
    }}, icon("copy", 18));
    return btn;
  };
  const save = h("a", {class: "icon-btn", "aria-label": "Download as text", title: "Download as text", download: name.replace(/\.zst$/, ".txt"), hidden: true}, icon("download", 20));
  const body = h("div", {class: "boot-body"}, h("p", {class: "muted"}, "Loading…"));
  let sections = [];
  const dump = (all) => sections.filter((s) => all || !s.el.hidden).map((s) => `==== ${s.title} ====\n${(all ? s.lines : s.shown).join("\n")}\n`).join("\n");
  const head = [filter, matches, copyBtn(() => dump(false)), save];
  const dialog = logDialog(name, head, body);
  // a block gets its marks as it comes into view: a common letter matches thousands of lines, more than a frame can mark
  const unmarked = new Map();  // block → [lines, q]
  const io = new IntersectionObserver((entries) => {
    for (const {target, isIntersecting} of entries) {
      const pending = unmarked.get(target);  // none for a block a later filter already replaced
      if (!isIntersecting || !pending) continue;
      const [lines, q] = pending;
      unmarked.delete(target);
      io.unobserve(target);
      target.replaceChildren(...lines.flatMap((l, j) => [j ? "\n" : "", ...marked(l, q)]));
    }
  }, {root: body, rootMargin: "50% 0px"});
  dialog.addEventListener("close", () => {
    ctl.abort();
    io.disconnect();
    URL.revokeObjectURL(save.href);
  });
  dialog.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "f" && !filter.hidden) {
      e.preventDefault();
      filter.focus();
      filter.select();
    }
  });
  // escape clears the filter before it closes the dialog
  dialog.addEventListener("cancel", (e) => {
    if (!filter.value) return;
    e.preventDefault();
    filter.value = "";
    apply();
  });

  let was = null;  // the sections' open state before filtering: the filter opens every section with a match
  function apply() {
    const q = filter.value.toLowerCase();
    if (q && !was) was = sections.map((s) => s.el.open);
    let n = 0;
    io.disconnect();
    unmarked.clear();
    sections.forEach((s, k) => {
      s.shown = render(s, q, (block, lines) => {
        unmarked.set(block, [lines, q]);
        io.observe(block);
      });
      s.el.hidden = !!q && !s.shown.length;
      if (q) s.el.open = true;
      else if (was) s.el.open = was[k];
      n += s.shown.length;
    });
    if (!q) was = null;
    matches.textContent = q ? `${n} line${n === 1 ? "" : "s"}` : "";
  }
  let timer = 0;
  filter.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(apply, 120);
  });

  readBoot(url, ctl.signal).then((b) => {
    if (ctl.signal.aborted) return;
    sections = bootSections(b).map(([title, text, open]) => {
      const s = {title, lines: text ? text.trimEnd().split("\n") : [], pre: h("pre", {class: "log-text"}), count: h("span", {class: "muted small"})};
      s.el = h("details", {class: "boot-section", open},
        h("summary", {}, h("span", {class: "mono"}, title), s.count, copyBtn(() => s.shown.join("\n"))), s.pre);
      return s;
    });
    apply();
    save.href = URL.createObjectURL(new Blob([dump(true)], {type: "text/plain"}));
    save.hidden = false;
    body.replaceChildren(bootSummary(b), ...sections.map((s) => s.el));
  }).catch((err) => {
    if (ctl.signal.aborted) return;
    for (const el of head.slice(0, 3)) el.hidden = true;
    // only the raw file is left to download, which may still be of use
    save.href = url;
    save.hidden = false;
    save.download = name;
    save.title = "Download";
    save.setAttribute("aria-label", "Download");
    body.replaceChildren(h("p", {class: "error"}, `Couldn't read this boot log: ${err.message}`));
  });
}

export function logsSection(dongleId, kind) {
  const el = h("div", {class: "section"}, h("p", {class: "muted"}, "Loading…"));
  (kind === "boot" ? api.bootlogs : api.crashlogs)(dongleId).then((urls) => {
    if (!urls.length) return el.replaceChildren(h("p", {class: "muted"}, `No ${kind} logs.`));
    const files = urls.map((url) => ({url, file: new URL(url).pathname.split("/").at(-1)})).reverse();
    const table = h("table", {class: kind === "boot" ? "list" : "list crashes"}, kind === "boot"
      ? h("tr", {}, h("th", {}, "boot"), h("th"))
      : h("tr", {}, h("th", {}, "time"), h("th", {}, "commit"), h("th", {}, "executable"), h("th")));
    const more = h("button", {class: "button", onclick: show}, "Show more");
    let shown = 0;
    function show() {
      for (const {url, file} of files.slice(shown, shown + PAGE)) {
        const cells = kind === "boot" ? [file.replace(/\.zst$/, "")] : crashName(file);
        const dl = h("a", {class: "dl", href: url, download: file, "aria-label": "Download", onclick: (e) => e.stopPropagation()}, icon("download", 18));
        // the whole row opens it; the name's a button for the keyboard, its click bubbling to the row
        table.append(h("tr", {class: "clickable", onclick: () => (kind === "boot" ? openBoot : openCrash)(url, file)},
          cells.map((c, i) => h("td", {class: i ? "mono muted" : "mono nowrap"}, i ? c : h("button", {class: "link mono"}, c))),
          h("td", {class: "right"}, dl)));
      }
      shown += PAGE;
      more.hidden = shown >= files.length;
    }
    show();
    el.replaceChildren(h("p", {class: "muted small"}, `${files.length} ${kind} logs, newest first`), table, h("p", {class: "center"}, more));
  }).catch((err) => el.replaceChildren(h("p", {class: "error"}, err.status === 403 ? "Only the device owner can see its logs." : String(err))));
  return el;
}
