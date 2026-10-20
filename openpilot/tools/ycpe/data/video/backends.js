// decoders behind one interface: WebCodecs (hardware first), else the wasm HEVC decoder. Each emits (n, frame),
// where frame is a VideoFrame on the GPU or {width, height, data} with tightly packed I420 planes for the shader.
const QUEUE = 4;  // chunks in flight in a WebCodecs decoder

export async function openBackend(codec, {emit, pump, error, alloc}) {
  for (const hardwareAcceleration of ["prefer-hardware", "no-preference"]) {
    const config = {codec, hardwareAcceleration, optimizeForLatency: true};
    if ((await VideoDecoder.isConfigSupported(config)).supported) return webcodecs(config, {emit, pump, error, alloc});
  }
  return codec.startsWith("hev") ? wasm({emit, pump, alloc}) : null;
}

function webcodecs(config, {emit, pump, error, alloc}) {
  const gpu = config.hardwareAcceleration === "prefer-hardware";
  let decoder = null;
  // software frames sit in CPU memory, where a texture upload of the VideoFrame is slow: pass the planes instead
  const output = async (frame) => {
    if (gpu) return emit(frame.timestamp, frame);
    const {width, height} = frame.visibleRect;
    const data = alloc(frame.allocationSize());
    await frame.copyTo(data);
    frame.close();
    emit(frame.timestamp, {width, height, data});
  };
  return {
    name: gpu ? "hardware" : "software",
    reset() {
      if (decoder?.state === "configured") {
        decoder.reset();
      } else {  // first use, or closed by a decode error
        decoder = new VideoDecoder({output, error: (e) => { console.error("VideoDecoder", e.message); error(); }});
        decoder.addEventListener("dequeue", pump);
      }
      decoder.configure(config);
    },
    ready: () => decoder.state === "configured" && decoder.decodeQueueSize < QUEUE,
    decode(data, key, n) { decoder.decode(new EncodedVideoChunk({type: key ? "key" : "delta", timestamp: n, data})); },
  };
}

// one access unit per task, so new requests get through between frames
async function wasm({emit, pump, alloc}) {
  // a build artifact (build.sh), not in git: without it there's no software HEVC (a static server may answer
  // with the app's index.html instead of a 404)
  const resp = await fetch(new URL("./hevc.wasm", import.meta.url));
  if (!resp.ok || resp.headers.get("content-type") !== "application/wasm") throw Object.assign(new Error(`hevc.wasm ${resp.status}`), {unbuilt: true});
  const module = await WebAssembly.compileStreaming(resp);
  // the libc imports left over (stdio, environ, clock) are never needed: each fails with ENOSYS
  const imports = {};
  for (const {module: m, name} of WebAssembly.Module.imports(module)) (imports[m] ??= {})[name] = () => 52;
  const w = (await WebAssembly.instantiate(module, imports)).exports;
  w._initialize?.();
  w.hevc_init();
  // yields between access units through a message, not setTimeout, whose nesting clamp adds 4 ms per frame
  let busy = false;
  const yielded = new MessageChannel();
  yielded.port1.onmessage = () => { busy = false; pump(); };
  const planes = () => {
    const width = w.hevc_width(), height = w.hevc_height();
    const cw = (width + 1) >> 1, ch = (height + 1) >> 1;
    const data = alloc(width * height + 2 * cw * ch);
    let off = 0;
    for (const [c, pw, ph] of [[0, width, height], [1, cw, ch], [2, cw, ch]]) {
      const src = new Uint8Array(w.memory.buffer, w.hevc_plane(c)), stride = w.hevc_stride(c);
      for (let y = 0; y < ph; y++, off += pw) data.set(src.subarray(y * stride, y * stride + pw), off);
    }
    return {width, height, data};
  };
  return {
    name: "wasm",
    reset() { w.hevc_reset(); },
    ready: () => !busy,
    decode(data, key, n) {
      const p = w.malloc(data.length);
      new Uint8Array(w.memory.buffer, p, data.length).set(data);
      w.hevc_push(p, data.length, n);
      w.free(p);
      while (w.hevc_frame()) {
        emit(w.hevc_pts(), planes());
        w.hevc_release();
      }
      busy = true;
      yielded.port2.postMessage(null);
    },
  };
}
