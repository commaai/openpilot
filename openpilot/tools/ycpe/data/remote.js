// the main thread's side of upstream-worker.js: one worker, started on first use, for everything fetched from GitHub
let worker = null, next = 0;
const calls = new Map();

export function upstream(op, args) {
  if (!worker) {
    worker = new Worker(new URL("./upstream-worker.js", import.meta.url), {type: "module"});
    worker.onmessage = ({data: {id, result, error}}) => {
      const call = calls.get(id);
      calls.delete(id);
      if (error) call.reject(new Error(error));
      else call.resolve(result);
    };
  }
  return new Promise((resolve, reject) => {
    const id = next++;
    calls.set(id, {resolve, reject});
    worker.postMessage({id, op, args});
  });
}
