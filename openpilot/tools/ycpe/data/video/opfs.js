// the on-disk cache in OPFS, shared with the logs: files keyed dongle/route/segment/name (the signed URLs change),
// trimmed to a budget by evicting the oldest-written files first, never the open route's
export const BUDGET = 10e9;

async function handle(key, create) {
  let dir = await navigator.storage.getDirectory();
  const parts = key.split("/");
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, {create});
  return dir.getFileHandle(parts.at(-1), {create});
}

// the cached file, or null
export async function read(key) {
  try {
    const f = await (await handle(key, false)).getFile();
    return f.size ? f : null;
  } catch {
    return null;
  }
}

// written beside the final name and moved into place, so a partial file is never read back
export async function write(key, bytes) {
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
    return true;
  } catch {
    return false;  // no OPFS, over quota, or another worker is writing it
  }
}

async function* files(dir, path = "") {
  for await (const [name, h] of dir.entries()) {
    if (h.kind === "directory") yield* files(h, `${path}${name}/`);
    else yield {key: path + name, dir, name, file: await h.getFile()};
  }
}

// drops the oldest files until the cache fits the budget; returns the keys removed
export async function evict(keep, budget = BUDGET) {
  const all = [];
  for await (const f of files(await navigator.storage.getDirectory())) all.push(f);
  let total = all.reduce((a, f) => a + f.file.size, 0);
  const removed = [];
  for (const f of all.sort((a, b) => a.file.lastModified - b.file.lastModified)) {
    if (total <= budget) break;
    if (f.key.startsWith(keep)) continue;
    await f.dir.removeEntry(f.name).catch(() => {});
    total -= f.file.size;
    removed.push(f.key);
  }
  return removed;
}
