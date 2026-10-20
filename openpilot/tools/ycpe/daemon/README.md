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
  header, or in `?tok=` for URLs the page can't set headers on (media elements, worker fetches). serve.py
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
| `POST /local/sources/resolve` | Body `{route, segs?, kinds?, sources?, mode?, sizes?}`. Returns `{route, segments: {n: {kind: {url, file, size, source, cached}}}, missing: {n: [kind]}, errors: {source: msg}}`. Details below. |
| `GET /local/file/{source}/{dongle}/{log_id}/{seg}/{file}` | The bytes, with Range support. It serves from the shared cache. On a miss it streams while the download fills the cache. A Range request on a miss is proxied, and the full download starts in the background. |
| `GET /local/cache[?route=&segs=]` | `{dir, budget, bytes, files, fills, hits, misses, fetched}`. With `route`, it adds per-segment `{kind: {file, source, size}}` of what's cached. |
| `POST /local/cache/prefetch` | Body as resolve. Starts a `prefetch` job (8 downloads at once). `meta.segs` gives each segment's state: queued, loading, disk or error. |
| `POST /local/cache/evict` | Body `{route, segs?, kinds?}`. Deletes the local cached copies. |
| `POST /local/cache/prune` | Body `{budget?}`. Applies the LRU budget now. |
| `GET /local/proxy?url=` | GET-only access to the hosts that send no CORS headers: commadataci, huggingface (and `*.hf.co` redirects), and the internal endpoint. Redirects are checked against the same allowlist, the response is always `application/octet-stream` with `CSP: sandbox`, and no credentials are added. |

Resolve parameters:
- `route` uses LogReader's syntax: `dongle|log_id[/slice][/q|r|a]`.
- `kinds` is a list of `rlog qlog qcamera fcamera ecamera dcamera log`. `log` means rlog, qlog, or rlog falling back to qlog for each segment, according to `mode`.
- `sources` is `"auto"` or a list. The auto order is device → local → internal → api → openpilotci → carSegments. A copy already in the shared cache wins over any download.
- The page's `Authorization: JWT` header, if sent, is used for the API source; otherwise `~/.comma/auth.json` is used.
- Each `url` already carries the token, so it can be fetched as-is.

## Shared cache

The cache lives in `Paths.download_cache_root()` (`/tmp/comma_download_cache/`). It's the directory cabana, jotpluggler, replay
(through `file_downloader.py`) and LogReader/FrameReader (through `url_file.py`) already use. Both formats key by the URL
without its query, so signed URLs of one file share an entry.

- **Lookup.** A file is looked up under the canonical URL of every source that could hold it, in any of three forms:
  - `sha256(url)`, the whole file
  - `sha256("decompressed-zst:" + url)`
  - complete `md5(url)_<n>.0` chunks, which are joined into the whole file
- **Writes.** Downloads land as `sha256(url)`. Logs are also mirrored as url_file chunks plus the manifest, so `LogReader`
  then reads them with no network.
- **Budget.** The whole directory is LRU-pruned on mtime down to `--cache-budget` (20 GB by default). Temporary files and
  anything touched in the last 2 minutes are skipped.
- **Daemon-made files.** Device pulls and other modules' outputs go under `ycpe/`:
  - `cache.derived(name)` returns a path for a module's output
  - `cache.fetch(dongle, log_id, seg, file, sources=None, job=None)` returns a local path, downloading into the cache if
    needed
