const kernelsReady = (async () => {
  // can't get browser to use updated versions except with cache-busting query string
  const exports = await import(`./net_clang.js?version=${Date.now()}`);
  Object.assign(self, exports);
})();

async function init(event) {
  await kernelsReady;
  self.model = await self.transformer();
  self.addEventListener("message", loadStateDict);
  self.removeEventListener("message", init);
  self.postMessage("success");
}

function loadStateDict(event) {
  if (event.data === "done") {
    self.addEventListener("message", inference);
    self.removeEventListener("message", loadStateDict);
  }
  else {
    if (event.data.length > 1) {
      // the bytes from files are set contiguously in WASM memory
      const malloc_size = event.data.reduce((sum, file) => sum + file.bytes.length, 0);
      const malloc_ptr = self.model.wasm._malloc(malloc_size);
      let cursor = 0;
      for (const file of event.data) {
        self.model.wasm.HEAPU8.set(file.bytes, malloc_ptr + cursor);
        for (const part of file.parts) {
          if (part.target_start_pos === 0) {
            // tell WASM code where the tensor is in memory
            self.model.wasm._set_buf(self.transformer_name_to_id[part.key], malloc_ptr + cursor);
          }
          cursor += part.size;
        }
        file.bytes = null;
      }
    }
    else {
      // the bytes from files are not guaranteed to be set contiguously in WASM memory
      const file = event.data[0];
      const malloc_ptr = self.model.wasm._malloc(file.size);
      self.model.wasm.HEAPU8.set(file.bytes, malloc_ptr);
      for (const part of file.parts) {
        if (part.target_start_pos === 0) {
          self.model.wasm._set_buf(self.transformer_name_to_id[part.key], malloc_ptr + part.file_start_pos);
        }
      }
      file.bytes = null;
    }
  }
  self.postMessage("success");
}

function inference(event) {
  const [tok, start_pos] = event.data;
  const int32tok = new Int32Array([tok]);
  const model_out = self.model.run(new Uint8Array(int32tok.buffer), start_pos);
  const int32nextTok = new Int32Array(model_out[0].buffer);
  self.postMessage(int32nextTok[0]);
}

self.addEventListener("message", init);