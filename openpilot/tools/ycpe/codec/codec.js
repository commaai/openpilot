// decompression for workers: zstd and bz2 in wasm, raw logs pass through
let wasm = null;

export async function initCodec(url = new URL("./codec.wasm", import.meta.url)) {
  if (wasm) return;
  const {instance} = await WebAssembly.instantiateStreaming(fetch(url), {env: {emscripten_notify_memory_growth() {}}});
  wasm = instance.exports;
  wasm._initialize();
}

const ZSTD_MAGIC = 0xfd2fb528, BZ2_MAGIC = 0x685a42;

// calls fn with the decompressed bytes, a view into wasm memory that is only valid during the call
export function withDecompressed(src, fn) {
  const magic = src.length >= 4 ? new DataView(src.buffer, src.byteOffset, 4).getUint32(0, true) : 0;
  const codec = magic === ZSTD_MAGIC ? wasm.zstd : (magic & 0xffffff) === BZ2_MAGIC ? wasm.bz2 : null;
  if (!codec) return fn(src);
  const p = wasm.malloc(src.length);
  new Uint8Array(wasm.memory.buffer, p, src.length).set(src);
  const err = codec(p, src.length);
  wasm.free(p);
  try {
    const out = new Uint8Array(wasm.memory.buffer, wasm.result(), wasm.result_len());
    if (err && !out.length) throw new Error(`decompress failed (${err})`);
    return fn(out);
  } finally {
    wasm.release();
  }
}

// returns a fresh Uint8Array that owns its buffer (transferable)
export const decompress = (src) => withDecompressed(src, (out) => out.slice());
