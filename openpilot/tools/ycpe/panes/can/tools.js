// CAN dialogs: DBC files, find similar bits, find signal, CSV export, route segments
import {h, ask, download, popover, menuList} from "/dom.js";
import {DBC} from "/data/dbc.js";
import {addrOf, carDbc, hexId} from "/data/can.js";
import {createSignal, dialog, field} from "./common.js";

const busList = (can) => [...new Set(can.messages().map((m) => m.bus))].sort((a, b) => a - b);

// open, save, copy and close DBCs, each for all buses or a chosen set; message is an error to show
export function dbcDialog(can, message = "") {
  const target = h("select", {}, h("option", {value: "-1"}, "all buses"), busList(can).map((b) => h("option", {value: b}, `bus ${b}`)));
  const buses = () => [+target.value];
  const error = h("div", {class: "error"}, message), note = h("div", {class: "muted"});
  // a file that opening or closing a DBC unloads drops its edits, so unsaved ones are confirmed first, as cabana
  const discard = (files) => {
    const dirty = files.filter((f) => can.dbcs.isDirty(f)).map((f) => f.name);
    return !dirty.length || ask(`${dirty.join(", ")} has unsaved changes. Discard them?`, "Discard");
  };
  const open = (fn) => async () => {
    if (!await discard(can.dbcs.replaced(buses()))) return;
    try {
      await fn();
      d.close();
    } catch (e) {
      // a cancelled file picker
      if (e.name !== "AbortError") error.textContent = e.message;
    }
  };

  const index = can.opendbc();
  const search = h("input", {placeholder: "Search opendbc", spellcheck: false, oninput: () => list()});
  const results = h("div", {class: "can-list"});
  async function list() {
    const {files, dbcsOf, label, local} = await index;
    const fp = [carDbc(dbcsOf(can.fingerprinted))].filter(Boolean);
    const q = search.value.toLowerCase();
    search.title = [local && `opendbc in ${local}`, label && `opendbc at ${label}`].filter(Boolean).join(", else ");
    results.replaceChildren(...(label || local ? [] : [h("div", {class: "muted"}, "Can't reach opendbc on GitHub")]),
      ...[...new Set([...fp, ...files])].filter((f) => f.includes(q)).map((f) =>
        h("button", {class: "can-list-row", onclick: open(() => can.loadOpendbc(f, buses()).then(() => remember(f, "opendbc")))},
          f, f === fp[0] && h("span", {class: "muted"}, " · this car"))));
  }
  list();

  const recents = recent();
  const recentList = recents.length ? h("div", {class: "can-list can-recent"}, h("div", {class: "muted"}, "Recent"),
    recents.map((r) => h("button", {class: "can-list-row", onclick: open(() => reopen(can, r, buses()))}, r.name, h("span", {class: "muted"}, ` · ${r.from}`)))) : null;

  // with the local daemon, DBCs also save into this machine's opendbc checkout
  const local = can.local();
  const localSave = (f) => {
    const slot = h("span");
    local.then((l) => l && slot.replaceWith(h("button", {class: "can-btn", title: `Write ${f.name}.dbc into ${l.dir}`,
      onclick: () => saveLocal(can, f, note, error)}, "Save to opendbc")));
    return slot;
  };

  const fail = (e) => { error.textContent = e.message; };
  const paste = h("textarea", {class: "can-paste", placeholder: "Paste DBC text", spellcheck: false});
  const files = h("div", {class: "can-files"}, can.dbcs.all().map((f) => h("div", {class: "can-file"},
    h("b", {}, f.name), h("span", {class: "muted"}, ` ${can.dbcs.buses(f).map((b) => (b < 0 ? "all buses" : `bus ${b}`)).join(", ")} · ${f.msgs.size} messages`),
    h("span", {class: "can-file-actions"},
      h("button", {class: "can-btn", onclick: () => save(can, f).catch(fail)}, "Save"),
      h("button", {class: "can-btn", onclick: () => save(can, f, true).catch(fail)}, "Save as"),
      localSave(f),
      h("button", {class: "can-btn", onclick: (e) => can.serialize(f).then((text) => navigator.clipboard.writeText(text)).then(() => { e.target.textContent = "Copied"; }, fail)}, "Copy"),
      h("button", {class: "can-btn", title: "Stop decoding with this DBC", onclick: async () => { if (await discard([f])) { can.dbcs.close(f); d.close(); } }}, "Unload")))));

  const d = dialog("DBC files", files,
    h("div", {class: "can-row"}, "Open for ", target),
    h("div", {class: "can-dbc-open"},
      h("div", {class: "can-col-flex"}, recentList, search, results),
      h("div", {class: "can-col-flex"},
        h("div", {class: "can-row"},
          h("button", {class: "can-btn", onclick: open(() => openFile(can, buses()))}, "Open file…"),
          h("button", {class: "can-btn", onclick: open(() => can.dbcs.open(buses(), new DBC("untitled", "")))}, "New")),
        paste, h("button", {class: "can-btn", onclick: open(async () => can.dbcs.open(buses(), await can.parseDbc("pasted", paste.value)))}, "Open pasted"))),
    error, note);
}

async function openFile(can, buses) {
  let handle, file;
  if (window.showOpenFilePicker) {
    [handle] = await showOpenFilePicker({types: [{description: "DBC", accept: {"text/plain": [".dbc"]}}]});
    file = await handle.getFile();
  } else {
    const input = h("input", {type: "file", accept: ".dbc"});
    file = await new Promise((resolve) => { input.onchange = () => resolve(input.files[0]); input.click(); });
  }
  const dbc = await can.parseDbc(file.name.replace(/\.dbc$/, ""), await file.text());
  dbc.handle = handle;
  can.dbcs.open(buses, dbc);
  if (handle) {
    await handles("readwrite", (store) => store.put(handle, dbc.name));
    remember(dbc.name, "file");
  }
}

// ---- recent DBCs, as cabana's Open Recent: names in localStorage, and opened files' handles in IndexedDB (only it
// can keep them)

const RECENT = "ycpe_can_recent";
const recent = () => JSON.parse(localStorage.getItem(RECENT) || "[]");

function remember(name, from) {
  const list = [{name, from}, ...recent().filter((r) => r.name !== name || r.from !== from)].slice(0, 10);
  localStorage.setItem(RECENT, JSON.stringify(list));
}

async function reopen(can, {name, from}, buses) {
  if (from === "opendbc") return can.loadOpendbc(name, buses);
  const handle = await handles("readonly", (store) => store.get(name));
  if (!handle || await handle.requestPermission({mode: "readwrite"}) === "denied") throw new Error(`${name}.dbc can't be reopened; open it again`);
  const dbc = await can.parseDbc(name, await (await handle.getFile()).text());
  dbc.handle = handle;
  can.dbcs.open(buses, dbc);
  remember(name, from);
}

// one IndexedDB request on the file handle store; resolves with its result
function handles(mode, fn) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("ycpe-can", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("handles");
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const q = fn(req.result.transaction("handles", mode).objectStore("handles"));
      q.onsuccess = () => resolve(q.result);
      q.onerror = () => reject(q.error);
    };
  });
}

// writes back to the opened file when permitted, else asks where (or downloads)
export async function save(can, dbc, as = false) {
  const text = await can.serialize(dbc);
  try {
    if (as || !dbc.handle) {
      if (!window.showSaveFilePicker) return download(text, `${dbc.name}.dbc`);
      dbc.handle = await showSaveFilePicker({suggestedName: `${dbc.name}.dbc`});
    }
    if (await dbc.handle.requestPermission?.({mode: "readwrite"}) === "denied") return download(text, `${dbc.name}.dbc`);
    const w = await dbc.handle.createWritable();
    await w.write(text);
    await w.close();
    can.dbcs.saved(dbc);
  } catch (e) {
    if (e.name !== "AbortError") throw e;
  }
}

// into the local opendbc; when the file changed on disk (or exists and wasn't read from there), asks first
async function saveLocal(can, dbc, note, error, force = false) {
  const r = await can.saveLocal(dbc, force).catch((e) => { error.textContent = e.message; });
  if (!r) return;
  if (r.conflict) {
    const buses = can.dbcs.buses(dbc);
    const d = dialog(`${dbc.name}.dbc changed on disk`,
      h("p", {}, dbc.local ? "It was modified since it was loaded." : "A file with this name is already in opendbc."),
      h("div", {class: "can-row"},
        h("button", {class: "can-btn", onclick: () => { d.close(); saveLocal(can, dbc, note, error, true); }}, "Overwrite"),
        dbc.local && h("button", {class: "can-btn", title: "Discards your edits", onclick: async () => {
          d.close();
          can.dbcs.close(dbc);
          await can.loadOpendbc(dbc.name, buses);
          note.textContent = `Reloaded ${dbc.name}.dbc`;
        }}, "Reload from disk"),
        h("button", {class: "can-btn", onclick: () => d.close()}, "Cancel")));
    return;
  }
  note.textContent = `Saved ${r.path}${r.generated ? " · a generated DBC: the generator will overwrite it, edit its source in generator/" : ""}`;
}

export function toolsMenu(can, ctx, anchor) {
  popover(anchor, () => menuList([
    {label: "Find similar bits…", fn: () => similarDialog(can)},
    {label: "Find signal…", fn: () => findDialog(can, ctx)},
    {label: "Export CSV…", fn: () => csvDialog(can, ctx)},
    {label: "Route info…", fn: () => segmentsDialog(ctx)}]));
}

const table = (cols, rows, onRow) => h("div", {class: "can-table"},
  h("div", {class: "can-table-head"}, cols.map((c) => h("span", {}, c))),
  rows.map((r) => h("div", {class: "can-table-row", ondblclick: () => onRow?.(r.src)}, r.cells.map((c) => h("span", {}, c)))));

function similarDialog(can) {
  const sel = can.selected ?? can.messages()[0]?.key;
  const msgs = can.messages().sort((a, b) => a.key - b.key);
  const src = h("select", {}, msgs.map((m) => h("option", {value: m.key, selected: m.key === sel}, `${hexId(m.key)} ${can.name(m.key)}`)));
  const byte = h("input", {type: "number", min: 0, max: 63, value: 0}), bit = h("input", {type: "number", min: 0, max: 7, value: 0});
  const bus = h("select", {}, busList(can).map((b) => h("option", {value: b}, b)));
  const equal = h("select", {}, h("option", {value: 1}, "Yes"), h("option", {value: 0}, "No"));
  const min = h("input", {type: "number", min: 0, value: 100});
  const out = h("div", {class: "can-results"});
  const find = async () => {
    out.textContent = "Searching…";
    const rows = await can.similarBits({k: +src.value, byte: +byte.value, bit: +bit.value, bus: +bus.value, equal: equal.value === "1", min: +min.value});
    out.replaceChildren(h("div", {class: "muted"}, `${rows.length} bits`), table(["Address", "Byte", "Bit", "Mismatches", "Messages", "Mismatched %"],
      rows.slice(0, 500).map((r) => ({src: r, cells: [addrOf(r.key).toString(16).toUpperCase(), r.byte, r.bit, r.mismatches, r.total, r.perc.toFixed(2)]})),
      (r) => can.select(r.key)));
  };
  dialog("Find similar bits",
    h("div", {class: "can-form"}, field("Find from", src), field("Byte index", byte), field("Bit index", bit),
      field("Find in bus", bus), field("Equal", equal), field("Minimum messages", min), h("button", {class: "can-btn primary", onclick: find}, "Find")),
    out).classList.add("can-wide");
}

function findDialog(can, ctx) {
  const inp = (value, attrs = {}) => h("input", {value, spellcheck: false, ...attrs});
  const f = {bus: inp("", {placeholder: "all"}), addr: inp("", {placeholder: "all, hex, comma separated"}), t0: inp("0"), t1: inp(ctx.clock.duration.toFixed(0)),
             min: inp("8", {type: "number", min: 1, max: 64}), max: inp("8", {type: "number", min: 1, max: 64}),
             le: h("input", {type: "checkbox", checked: true}), signed: h("input", {type: "checkbox"}), factor: inp("1"), offset: inp("0"),
             cmp: h("select", {}, ["=", ">", ">=", "!=", "<", "<=", "between"].map((c) => h("option", {}, c))), v1: inp("0"), v2: inp("0")};
  const status = h("span", {class: "muted"});
  const out = h("div", {class: "can-results"});
  const setup = [f.bus, f.addr, f.t0, f.t1, f.min, f.max, f.le, f.signed, f.factor, f.offset];
  let started = false;
  const list = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
  const show = (hits, total) => {
    status.textContent = `${total} matches`;
    out.replaceChildren(table(["ID", "Start bit, size", "(time, value)"], hits.map((c) => ({src: c, cells: [hexId(c.key), `${c.sig.start}, ${c.sig.size}`,
      c.values.map(([t, v]) => `(${t.toFixed(3)}, ${+v.toPrecision(6)})`).join(" ")]})), (c) => {
      can.select(c.key);
      createSignal(can, c.key, c.sig);
    }));
  };
  async function next() {
    if (!started) {
      await can.findInit({buses: list(f.bus.value).map(Number), addresses: list(f.addr.value).map((x) => parseInt(x, 16)),
                          t0: +f.t0.value, t1: +f.t1.value || Infinity, minSize: +f.min.value, maxSize: +f.max.value,
                          le: f.le.checked, signed: f.signed.checked, factor: +f.factor.value, offset: +f.offset.value});
      started = true;
      setup.forEach((x) => { x.disabled = true; });
    }
    status.textContent = "Searching…";
    const hits = await can.findNext(f.cmp.value, +f.v1.value, +f.v2.value);
    show(hits, await can.call("findCount"));
    findBtn.textContent = "Find next";
  }
  const findBtn = h("button", {class: "can-btn primary", onclick: next}, "Find");
  dialog("Find signal",
    h("div", {class: "can-form can-form-2"},
      h("div", {}, h("b", {}, "Messages"), field("Bus", f.bus), field("Address", f.addr), field("Time", h("span", {class: "can-row"}, f.t0, "–", f.t1, "s"))),
      h("div", {}, h("b", {}, "Signal"), field("Size", h("span", {class: "can-row"}, f.min, "–", f.max)),
        h("label", {class: "can-check"}, f.le, "Little endian"), h("label", {class: "can-check"}, f.signed, "Signed"), field("Factor", f.factor), field("Offset", f.offset))),
    h("div", {class: "can-row"}, "Value ", f.cmp, f.v1, f.v2, findBtn,
      h("button", {class: "can-btn", onclick: async () => show(await can.findUndo(), await can.call("findCount"))}, "Undo"),
      h("button", {class: "can-btn", onclick: () => { started = false; setup.forEach((x) => { x.disabled = false; }); findBtn.textContent = "Find"; out.replaceChildren(); status.textContent = ""; }}, "Reset"),
      status),
    h("div", {class: "muted"}, "Double-click a match to create the signal."),
    out).classList.add("can-wide");
}

function csvDialog(can, ctx) {
  const sel = can.selected;
  const [t0, t1] = ctx.clock.range;
  const range = h("select", {}, h("option", {value: "all"}, "whole route"), h("option", {value: "view"}, `visible range (${t0.toFixed(1)}–${t1.toFixed(1)} s)`));
  const what = h("select", {},
    h("option", {value: "all"}, "raw frames, all messages"),
    sel !== null && h("option", {value: "raw", selected: true}, `raw frames, ${hexId(sel)}`),
    sel !== null && can.dbcMsg(sel) && h("option", {value: "decoded"}, `decoded signals, ${can.name(sel)}`));
  const d = dialog("Export CSV", h("div", {class: "can-form"}, field("Data", what), field("Range", range),
    h("button", {class: "can-btn primary", onclick: async () => {
      const [a, b] = range.value === "all" ? [-Infinity, Infinity] : [t0, t1];
      const keys = what.value === "all" ? can.messages().map((m) => m.key) : [sel];
      const blob = await can.csv(keys, a, b, what.value === "decoded");
      download(blob, `${ctx.route.fullname.replace("|", "_")}_${what.value === "all" ? "can" : hexId(sel).replace(":", "_")}.csv`);
      d.close();
    }}, "Export")));
}

// cabana's route info: the fingerprint and each segment; a row seeks there (the Files view lists every file)
function segmentsDialog(ctx) {
  const segs = ctx.store.route.segments, fp = ctx.store.valueAt("/carParams/carFingerprint", Infinity);
  dialog("Route info", h("div", {class: "muted"}, ctx.route.fullname, fp ? ` · ${fp}` : ""),
    table(["Segment", "Start", "End", "Log"], segs.map((s) => ({src: s, cells: [s.n, s.t0.toFixed(1), s.t1.toFixed(1), s.status]})), (s) => ctx.clock.seek(s.t0)));
}
