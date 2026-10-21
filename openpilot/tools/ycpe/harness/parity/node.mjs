// what the node parity scripts share: the app's decoding (codec.wasm, bundled schema)
import fs from "node:fs";
import {initCodec} from "../../codec/codec.js";

export const schema = JSON.parse(fs.readFileSync(new URL("../../data/schema.json", import.meta.url)));

// initCodec fetches codec.wasm, so file: URLs read from disk
const netFetch = globalThis.fetch;
globalThis.fetch = async (u, o) => String(u).startsWith("file:") ? new Response(fs.readFileSync(new URL(u)), {headers: {"content-type": "application/wasm"}}) : netFetch(u, o);
await initCodec();
