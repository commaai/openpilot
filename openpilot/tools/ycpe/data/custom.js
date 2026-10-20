// custom series: user JavaScript evaluated per sample in a worker of its own. That worker is same-origin, not a
// sandbox: the code can do what this page can, so code from an imported layout runs only once reviewed. Inputs are
// store paths; every input after the first is resampled (sample-and-hold) onto the first one's times. The globals
// block runs once per evaluation, so its variables persist across samples. The function body sees time, value, v1 … vN.
import {Feed, receive, resample, leases} from "./transforms.js";

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

if (globalThis.WorkerGlobalScope && self.location.href === import.meta.url) {
  onmessage = ({data: msg}) => {
    const series = receive(msg);
    if (!series) return;
    try {
      const {inputs, globals, code} = msg, [first, ...rest] = inputs.map((p) => series.get(p));
      const extra = rest.map((s) => resample(first.t, s.t, s.v));
      const fn = compile(inputs, globals, code), t = first.t.slice(), v = new Float64Array(t.length), args = new Array(2 + extra.length);
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

// one worker runs the jobs in turn, holding their inputs' chunks so each crosses threads once. A job that runs
// past TIMEOUT_MS takes the worker down; the jobs queued behind it go to a new one, and the same code fails at once
// on that store from then on, so it doesn't hold the others up again as its inputs change
let worker = null, feed = null, timer = 0;
const jobs = [];  // posted and unanswered, the running one first: {store, path, spec, resolve, reject}
const hung = new WeakMap();  // store → the specs that timed out on it
const TIMED_OUT = `timed out after ${TIMEOUT_MS / 1000} s`;
const key = ({inputs, globals, code}) => JSON.stringify([inputs, globals, code]);
const timedOut = ({store, spec}) => !!hung.get(store)?.has(key(spec));
// the inputs each pane's custom series read (see transforms.js leases())
export const lease = leases(() => feed);

function post(job) {
  if (!worker) {
    worker = new Worker(import.meta.url, {type: "module"});
    worker.onmessage = ({data: r}) => finish(r.error ? new Error(r.error) : r);
    worker.onerror = (e) => restart(new Error(e.message));
    feed = new Feed(worker);
  }
  const transfer = [];
  try {
    worker.postMessage({...job.spec, ...feed.feed(job.store, job.spec.inputs, transfer)}, transfer);
  } catch (e) {
    return job.reject(e);
  }
  if (jobs.push(job) === 1) arm();
}

function arm() {
  clearTimeout(timer);
  if (!jobs.length) return;
  timer = setTimeout(() => {
    const {store, spec} = jobs[0];
    if (!hung.has(store)) hung.set(store, new Set());
    hung.get(store).add(key(spec));
    restart(new Error(TIMED_OUT));
  }, TIMEOUT_MS);
}

function finish(r) {
  const job = jobs.shift();
  arm();
  if (r instanceof Error) return job.reject(r);
  job.store.put(job.path, {type: "f64", t: r.t, v: r.v});
  job.resolve(job.path);
}

function restart(error) {
  worker.terminate();
  worker = null;
  const [job, ...rest] = jobs.splice(0);
  job?.reject(error);
  for (const j of rest) timedOut(j) ? j.reject(new Error(TIMED_OUT)) : post(j);
  arm();
}

// evaluates a custom series into the store at path
export function evaluate(store, path, {inputs, globals = "", code}) {
  const spec = {inputs, globals, code};
  if (timedOut({store, spec})) return Promise.reject(new Error(TIMED_OUT));
  return new Promise((resolve, reject) => post({store, path, spec, resolve, reject}));
}
