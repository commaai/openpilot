// QR codes read off the main thread where the platform has no reader (Chrome on Linux and Windows), with the
// BarcodeDetector ponyfill connect uses: zxing-cpp in wasm, ~8 ms a 720p frame. Both files are pinned by hash.
const JS = "https://cdn.jsdelivr.net/npm/barcode-detector@3.2.3/dist/iife/ponyfill.js";
const JS_SRI = "sha384-ZQvvHKgw7Ktqkui5vapXokyfDpQcnXLZhR9hfFMh8rUQztTRNywKQdQ994308nhY";
const WASM = "https://cdn.jsdelivr.net/npm/zxing-wasm@3.1.6/dist/reader/zxing_reader.wasm";  // the version the ponyfill's built for
const WASM_SRI = "sha384-nK2pHJVglp+a6HfEzYu3Das+oIISIChcnPvLUzhPoPHhi8NaJzy9dzcwIdK/dC0I";

const detector = Promise.all([
  fetch(JS, {integrity: JS_SRI}).then((r) => r.blob()),
  fetch(WASM, {integrity: WASM_SRI}).then((r) => r.arrayBuffer()),
]).then(([js, wasm]) => {
  importScripts(URL.createObjectURL(js));
  const {BarcodeDetector, setZXingModuleOverrides} = self.BarcodeDetectionAPI;
  setZXingModuleOverrides({wasmBinary: wasm});
  return new BarcodeDetector({formats: ["qr_code"]});
});

detector.then(() => postMessage({ready: true}), (e) => postMessage({error: e.message}));

// a frame in, the codes' text out
onmessage = async ({data: frame}) => {
  try {
    const codes = await (await detector).detect(frame);
    postMessage({codes: codes.map((c) => c.rawValue)});
  } catch (e) {
    postMessage({error: e.message});
  } finally {
    frame.close();
  }
};
