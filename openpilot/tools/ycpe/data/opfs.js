// the on-disk cache in OPFS for logs and video, read from the page or any worker and written from dedicated workers
// (writes need a sync access handle). Files are keyed by route, segment and name (the signed URLs change). One LRU
// index in IndexedDB records every file's kind, size and last use, and each kind is trimmed to its share of the
// budget, oldest use first, never touching the open route.
export const SHARES = {logs: 5e9, video: 10e9};  // bytes; setShare() overrides, per origin

async function handle(key, create) {
  let dir = await navigator.storage.getDirectory();
  const parts = key.split("/");
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, {create});
  return dir.getFileHandle(parts.at(-1), {create});
}

// the index: "files" maps key → {kind, bytes, used}, "shares" maps kind → bytes
let db = null;
function open() {
  return db ||= new Promise((resolve, reject) => {
    const r = indexedDB.open("ycpe-opfs", 1);
    r.onupgradeneeded = () => {
      r.result.createObjectStore("files");
      r.result.createObjectStore("shares");
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function idb(store, mode, fn) {
  const tx = (await open()).transaction(store, mode), req = fn(tx.objectStore(store));
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = () => reject(tx.error);
  });
}

const touch = (key, entry) => idb("files", "readwrite", (s) => s.put({...entry, used: Date.now()}, key)).catch(() => {});

// the cached file (a File), or null; a read counts as a use
export async function read(key) {
  try {
    const f = await (await handle(key, false)).getFile();
    if (!f.size) return null;
    const entry = await idb("files", "readonly", (s) => s.get(key));
    if (entry) touch(key, entry);
    return f;
  } catch {
    return null;
  }
}

// written beside the final name and moved into place, so a partial file is never read back; false if it couldn't be
export async function write(key, bytes, kind) {
  try {
    const part = await handle(key + ".part", true), h = await part.createSyncAccessHandle();
    try {
      h.truncate(0);
      h.write(bytes, {at: 0});
      h.flush();
    } finally {
      h.close();
    }
    await part.move(key.split("/").at(-1));
    await touch(key, {kind, bytes: bytes.byteLength});
    return true;
  } catch {
    return false;  // no OPFS, over quota, or another worker is writing it
  }
}

// {key: bytes} of the cached files under a key prefix
export async function list(prefix) {
  const out = {};
  const range = IDBKeyRange.bound(prefix, prefix + "￿");
  await idb("files", "readonly", (s) => {
    const req = s.openCursor(range);
    req.onsuccess = () => {
      const c = req.result;
      if (!c) return;
      out[c.key] = c.value.bytes;
      c.continue();
    };
  });
  return out;
}

export const share = async (kind) => (await idb("shares", "readonly", (s) => s.get(kind)).catch(() => null)) ?? SHARES[kind];
export const setShare = (kind, bytes) => idb("shares", "readwrite", (s) => s.put(bytes, kind));

// trims a kind to its share, least recently used first, skipping any key under the open route (dongle/route);
// returns the keys removed
export async function evict(kind, route) {
  const budget = await share(kind), files = [];
  await idb("files", "readonly", (s) => {
    const req = s.openCursor();
    req.onsuccess = () => {
      const c = req.result;
      if (!c) return;
      if (c.value.kind === kind) files.push([c.key, c.value]);
      c.continue();
    };
  });
  let total = files.reduce((a, [, f]) => a + f.bytes, 0);
  const removed = [];
  for (const [key, f] of files.sort((a, b) => a[1].used - b[1].used)) {
    if (total <= budget) break;
    if (key.includes(`${route}/`)) continue;
    const parts = key.split("/");
    try {
      let dir = await navigator.storage.getDirectory();
      for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p);
      await dir.removeEntry(parts.at(-1));
    } catch {}  // already gone
    await idb("files", "readwrite", (s) => s.delete(key));
    total -= f.bytes;
    removed.push(key);
  }
  return removed;
}
