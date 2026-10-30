// calls into a worker: call(op, args, transfer) posts {id, op, ...args} and resolves with the reply's result. A worker
// that doesn't load, or a reply that can't be read, fails every call waiting and every later one, with the reason
// fail(event) gives. Other messages (no id of a call) are left to the worker's other listeners
export function client(worker, fail = (e) => e.message || `the worker failed: ${e.type}`) {
  const calls = new Map();
  let next = 0, failed = null;
  worker.onmessage = ({data: {id, result, error, stack}}) => {
    const c = calls.get(id);
    calls.delete(id);
    if (error) c?.reject(Object.assign(new Error(error), stack && {stack})); else c?.resolve(result);
  };
  worker.onerror = worker.onmessageerror = (e) => {
    failed ??= fail(e);
    for (const c of calls.values()) c.reject(new Error(failed));
    calls.clear();
  };
  return (op, args = {}, transfer = []) => failed !== null ? Promise.reject(new Error(failed)) : new Promise((resolve, reject) => {
    const id = next++;
    calls.set(id, {resolve, reject});
    worker.postMessage({id, op, ...args}, transfer);
  });
}

// the worker's side: ops[op](args, transfer) answers each call, pushing the buffers its result hands over to transfer
export function serve(ops) {
  globalThis.onmessage = async ({data: {id, op, ...args}}) => {
    const transfer = [];
    try {
      postMessage({id, result: await ops[op](args, transfer)}, transfer);
    } catch (e) {
      postMessage({id, error: e.message || String(e), stack: e.stack});
    }
  };
}
