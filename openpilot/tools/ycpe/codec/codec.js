// decompression for workers: zstd and bz2 in wasm, raw logs pass through
let wasm = null;

export async function initCodec(url = new URL("./codec.wasm", import.meta.url)) {
  if (wasm) return;
  const {instance} = await WebAssembly.instantiateStreaming(fetch(url), {env: {emscripten_notify_memory_growth() {}}});
  wasm = instance.exports;
  wasm._initialize();
}

const ZSTD_MAGIC = 0xfd2fb528, BZ2_MAGIC = 0x685a42;

// returns a fresh Uint8Array that owns its buffer (transferable)
export function decompress(src) {
  const magic = src.length >= 4 ? new DataView(src.buffer, src.byteOffset, 4).getUint32(0, true) : 0;
  const fn = magic === ZSTD_MAGIC ? wasm.zstd : (magic & 0xffffff) === BZ2_MAGIC ? wasm.bz2 : null;
  if (!fn) return src.slice();
  const p = wasm.malloc(src.length);
  new Uint8Array(wasm.memory.buffer, p, src.length).set(src);
  const err = fn(p, src.length);
  wasm.free(p);
  const out = new Uint8Array(wasm.memory.buffer, wasm.result(), wasm.result_len()).slice();
  wasm.release();
  if (err && !out.length) throw new Error(`decompress failed (${err})`);
  return out;
}
