// serves /p2p/<page>/<dongle>/<log id>--<n>/<file>: a device's file read peer to peer by the page that made the URL
// (data/live/p2p.js), so the segment workers fetch, resume and cache it in OPFS like any download. Every other request
// goes to the network untouched. It lives at the root because a service worker only controls pages under it.
const ANSWER_MS = 10000;  // for the page to start the read; past it, a 503 sends the worker into its retries

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin === location.origin && url.pathname.startsWith("/p2p/")) e.respondWith(serve(url, e.request));
});

const fail = (status, error) => new Response(JSON.stringify({error}), {status, headers: {"Content-Type": "application/json"}});

async function serve(url, req) {
  const [page, dongle, ...path] = url.pathname.slice("/p2p/".length).split("/").map(decodeURIComponent);
  const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.get("range") || "");
  const offset = m ? +m[1] : 0, length = m?.[2] ? +m[2] + 1 - offset : undefined;
  const ask = {p2p: {page, dongle, path: path.join("/"), offset, length}};
  // every window of the app hears the ask on a port of its own; the page with this id answers {size} or {error},
  // then the bytes as {chunk} messages and {done}
  return new Promise((resolve) => {
    const ports = [];
    const timer = setTimeout(() => settle(null, fail(503, "no page is reading this device's files")), ANSWER_MS);
    const settle = (winner, response) => {
      clearTimeout(timer);
      for (const p of ports) if (p !== winner) p.close();
      resolve(response);
    };
    self.clients.matchAll({type: "window"}).then((windows) => {
      for (const w of windows) {
        const {port1, port2} = new MessageChannel();
        ports.push(port1);
        port1.onmessage = ({data}) => {
          if (data.error) return settle(null, fail(data.status || 502, data.error));
          settle(port1, new Response(body(port1), {
            status: m ? 206 : 200,
            headers: {
              "Content-Type": "application/octet-stream",
              "Content-Length": String(length ?? data.size - offset),
              "Accept-Ranges": "bytes",
              ...(m ? {"Content-Range": `bytes ${offset}-${offset + (length ?? data.size - offset) - 1}/${data.size}`} : {}),
            },
          }));
        };
        w.postMessage(ask, [port2]);
      }
    });
  });
}

// the bytes that follow on port; cancelling the response (an aborted fetch) cancels the read on the device
const body = (port) => new ReadableStream({
  start(ctl) {
    port.onmessage = ({data}) => {
      if (data.chunk) ctl.enqueue(new Uint8Array(data.chunk));
      else if (data.done) ctl.close();
      else if (data.error) ctl.error(new Error(data.error));
      if (data.done || data.error) port.close();
    };
  },
  cancel() {
    port.postMessage({cancel: true});
    port.close();
  },
});
