# ycpe local daemon

`serve.py` serves the app on 127.0.0.1 and, when run with openpilot's venv, the `/local/*` API that unlocks
what a browser can't do alone: your `~/.comma` sign-in, log sources without CORS, the download cache shared
with the Python and C++ tools, local repos, and pulling logs from a device. The page finds it through
`<meta name="ycpe-daemon">` in index.html plus `GET /local/caps`. Without it everything still works
(`serve.py --static`, or any static host), and the features that need it say so.

```bash
PYTHONPATH=$PWD .venv/bin/python openpilot/tools/ycpe/serve.py            # http://localhost:3000/ (athena wants :3000)
PYTHONPATH=$PWD .venv/bin/python -m unittest openpilot.tools.ycpe.daemon.tests.test_security
```

## Security

- It binds 127.0.0.1 only.
- Every request, static files included, must carry `Host: localhost:<port>` or `127.0.0.1:<port>`. This
  stops DNS rebinding.
- `/local/*` rejects any `Origin` other than the daemon's own, and any `Sec-Fetch-Site` other than
  `same-origin` or `none`. It never sends CORS headers.
- Every `/local/*` route except `caps` needs the per-process session token. It goes in the `X-Ycpe-Token`
  header, or in `?t=` for URLs the page can't set headers on (media elements, worker fetches). serve.py
  puts the token in index.html, and the request log never prints query strings.
- File reads go through `confine(path, roots)`, which resolves symlinks and rejects `..`.
- The proxy only reaches an allowlist of hosts.

## Writing a module

Add `daemon/<name>.py` and list it in `server.MODULES`. A module that fails to import is reported under
`failed` in `/local/caps`, and everything else keeps working.

```python
from openpilot.tools.ycpe.daemon import File, HTTPError, Stream, config, confine, module, route, sse
from openpilot.tools.ycpe.daemon import jobs

module("transcode", "1", presets=["720p"])   # listed in /local/caps → {"modules": {"transcode": {"version": "1", "presets": [...]}}}

@route("GET", "/local/transcode/{route}/{seg}/{cam}")   # {x} = one path segment, {x:path} = the rest
def get(req):
  req.params["seg"]; req.query.get("preset"); req.arg("n", 0, int); req.json(); req.headers["range"]
  return {"ok": 1}                       # dict/list → JSON; None → 204; raise HTTPError(404, "why") for errors
  return File(path)                      # Range, 206, sendfile
  return File(path, growing=lambda: job.state == "running")  # chunked, follows the tail until growing() is false
  return Stream(gen(), "video/mp4")      # chunked; gen.close() runs when the client goes away
  return sse(iter_of_event_data_pairs)   # text/event-stream; yield None for a keepalive
```

Handlers run in the server's request threads, so long work belongs in a job:

```python
job = jobs.start("transcode", fn, title="fcamera 3 → 720p", route=..., seg=3)   # extra kwargs land in job.meta
def fn(job):
  job.progress(done, total, text="12 MB/s", unit="bytes")   # throttled to 10 Hz for listeners
  job.check()                                               # raises Cancelled once cancel was asked for
  return {"path": ...}                                      # job.result
```

| endpoint | |
|---|---|
| `GET /local/jobs` | every job: `{id, kind, title, state: running\|done\|error\|cancelled, done, total, unit, text, result, error, started, ended, meta}` |
| `GET /local/jobs/{id}` | one job |
| `POST /local/jobs/{id}/cancel` | asks the job to stop |
| `GET /local/jobs/events` | SSE `event: job`, the current jobs on connect and then every update |

On the page, `daemon.js` exports `daemon.{present, caps, has(module), fetch(path, opts), json(path, opts),
url(path), events(path, fn)}` and `needsDaemon(what)`.

## Endpoints

| endpoint | |
|---|---|
| `GET /local/caps` | `{daemon, version, modules: {name: {version, ...}}, failed: {name: error}}`. This is the only route that needs no token. |
| `GET /local/auth` | `{token}`: the JWT from `~/.comma/auth.json`, or null. |
