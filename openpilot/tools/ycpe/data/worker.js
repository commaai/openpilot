// segment pipeline worker: fetch → decompress → read → compact → transfer back to the loader
import {initCodec, decompress} from "../codec/codec.js";
import {Reader, compact} from "./capnp.js";

const ready = Promise.all([
  initCodec(),
  fetch(new URL("./schema.json", import.meta.url)).then((r) => r.json()).then((s) => new Reader(s)),
]).then(([, reader]) => reader);

onmessage = async ({data: {id, url}}) => {
  try {
    const reader = await ready;
    const t0 = performance.now();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${resp.status}`);
    const src = new Uint8Array(await resp.arrayBuffer());
    const t1 = performance.now();
    const raw = decompress(src);
    const t2 = performance.now();
    const {services, transfer} = compact(reader.read(raw));
    const t3 = performance.now();
    const ms = {fetch: t1 - t0, decompress: t2 - t1, read: t3 - t2};
    postMessage({id, services, bytes: src.byteLength, raw: raw.byteLength, ms}, transfer);
  } catch (e) {
    postMessage({id, error: String(e)});
  }
};
