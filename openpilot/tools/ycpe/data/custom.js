// custom series: user JavaScript evaluated per sample in a throwaway worker. Inputs are store paths; every
// input after the first is resampled (sample-and-hold) onto the first one's times. The globals block runs
// once, so its variables persist across samples. The function body sees time, value, v1 … vN.
import {concat, resample} from "./transforms.js";

export const TEMPLATES = {
  Derivative: {globals: "let pt = NaN, pv = NaN;", code: "const d = (value - pv) / (time - pt);\npt = time;\npv = value;\nreturn d;"},
  Difference: {inputs: 2, globals: "", code: "return value - v1;"},
  Smoothing: {globals: "const n = 20;\nconst buf = [];\nlet sum = 0;", code: "buf.push(value);\nsum += value;\nif (buf.length > n) sum -= buf.shift();\nreturn sum / buf.length;"},
  Integral: {globals: "let acc = 0, pt = NaN;", code: "if (pt === pt) acc += value * (time - pt);\npt = time;\nreturn acc;"},
};

const TIMEOUT_MS = 10000;

export function compile(inputs, globals, code) {
  const args = ["time", "value", ...inputs.slice(1).map((_, i) => `v${i + 1}`)];
  return new Function(`"use strict";\n${globals}\nreturn function (${args.join(", ")}) {\n${code}\n};`)();
}

if (typeof window === "undefined" && self.location.href === import.meta.url) {
  onmessage = ({data: {inputs, globals, code, data}}) => {
    try {
      const [first, ...rest] = inputs.map((p) => concat(data[p]));
      const extra = rest.map((s) => resample(first.t, s.t, s.v));
      const fn = compile(inputs, globals, code), t = first.t, v = new Float64Array(t.length), args = new Array(2 + extra.length);
      for (let i = 0; i < t.length; i++) {
        args[0] = t[i];
        args[1] = first.v[i];
        for (let k = 0; k < extra.length; k++) args[2 + k] = extra[k][i];
        v[i] = +fn(...args);
      }
      postMessage({t, v}, [t.buffer, v.buffer]);
    } catch (e) {
      postMessage({error: `${e.name}: ${e.message}`});
    }
  };
}

// evaluates a custom series into the store at path
export function evaluate(store, path, {inputs, globals = "", code}) {
  return new Promise((resolve, reject) => {
    const data = {};
    for (const p of inputs) {
      const s = store.series(p);
      if (!s) return reject(new Error(`no series ${p}`));
      data[p] = s.chunks;
    }
    const w = new Worker(import.meta.url, {type: "module"});
    const timer = setTimeout(() => { w.terminate(); reject(new Error(`timed out after ${TIMEOUT_MS / 1000} s`)); }, TIMEOUT_MS);
    w.onmessage = ({data: r}) => {
      clearTimeout(timer);
      w.terminate();
      if (r.error) return reject(new Error(r.error));
      store.put(path, {type: "f64", t: r.t, v: r.v});
      resolve(path);
    };
    w.onerror = (e) => { clearTimeout(timer); w.terminate(); reject(new Error(e.message)); };
    w.postMessage({inputs, globals, code, data});
  });
}
