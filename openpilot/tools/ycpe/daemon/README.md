# ycpe local daemon

ycpe is a static web app, and works as one. Run it through `serve.py` with openpilot's Python and you also get a
small local server that does what a browser can't on its own.

## Using it

```bash
.venv/bin/python openpilot/tools/ycpe/serve.py          # open http://localhost:3000/
```

- **Port.** Use port 3000. The default is the one athena (device RPC from the browser) allows.
- **Static only.** `--static`, or running without openpilot's venv, gives the plain static app.
- **Status.** A **local** badge in the top bar means the daemon is on. Click it for the disk cache, the
  device, and running jobs.

| Option | Default | |
|---|---|---|
| `--cache-budget GB` | 20 | Size of the shared download cache, pruned least-recently-used first. |
| `--device HOST` | `$YCPE_DEVICE` | A comma device on your network. The ssh user defaults to `comma`. |
| `--device-key FILE` | `$YCPE_DEVICE_KEY` | The ssh key for the device. |
| `--data-dir DIR` | | A folder of route segments on disk (`<log id>--<n>/rlog.zst`, as on a device). It can be repeated. |
| `--repo DIR` | | An openpilot checkout. Defaults to the one `serve.py` is in. |

What the daemon unlocks in the UI:

| Module | What you get |
|---|---|
| `auth` | Your `~/.comma/auth.json` sign-in (from `tools/lib/auth.py`). The app opens signed in, and the account menu says so. |
| `sources` | Logs from every source openpilot's LogReader uses: your disk, a pulled device, comma's internal storage, the comma API, openpilotci and commaCarSegments. That includes routes the browser can't reach. In the Files pane, the source picker enables these. |
| `cache` | One disk cache shared with cabana, jotpluggler, replay and LogReader, so a route downloaded by any of them loads instantly in all of them. The badge panel can cache a whole route ahead of time (logs or video) and evict it. Progress shows in the loading pill. |
| `repos` | The local cereal schema (for a "local checkout" schema), and opendbc DBCs. The CAN view can save DBCs straight back into your opendbc checkout. |
| `device` | Lists the routes on your comma device and copies them over ssh into the cache. Nothing on the device is ever changed. |
| `proxy` | Reaches log hosts that browsers can't (openpilotci, huggingface). |
| `jobs` | Progress and cancel for anything long, shown in the badge panel. |
| `transcode`, `replay` | Video transcoding and openpilot re-runs, when installed (see their modules). |

Without the daemon these features are disabled, with a short note saying they need it.

**Is it safe?**
1. The daemon only listens on 127.0.0.1, and it refuses requests addressed to any other host name, which blocks DNS rebinding.
2. Every API call needs a random token that only the page `serve.py` served carries. Other websites can't get it, because
   cross-origin requests get no CORS headers and are rejected, and no other site may frame the app.
3. It reads files only from the cache, your data dirs and your checkouts, and it writes only to the cache and to opendbc's DBC folder.
4. It never modifies anything on a comma device or in your comma account. The only commands it runs on a device are `cat` and `find`.
5. Your JWT and the session token are never logged.

## For developers

```bash
PYTHONPATH=$PWD .venv/bin/python -m unittest discover -s openpilot/tools/ycpe/daemon/tests -t .   # device tests need ssh to localhost
```

The page finds the daemon through `<meta name="ycpe-daemon">` in index.html, plus `GET /local/caps`.

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
| `GET /local/repos` | `[{id, name, path, git: {branch, commit, subject, dirty}, cereal, opendbc: {dbc, git}}]`. These are the `--repo` checkouts (the running one by default), plus a worktree's main checkout. |
| `GET /local/repos/{id}/capnp` | `{files: {import path: text}}`: cereal's `*.capnp` (including `include/c++.capnp`) and opendbc's `car.capnp`. |
| `GET /local/repos/{id}/dbc` | `[{name, size, mtime}]` |
| `GET /local/repos/{id}/dbc/{name}` | `{name, text, mtime}` |
| `PUT /local/repos/{id}/dbc/{name}[?mtime=]` | Body is the DBC text. Saves it atomically into opendbc's dbc directory, and only there: the name must match `[A-Za-z0-9_.-]+\.dbc`, and the path is confined. With the `mtime` from the read, it answers 409 if the file changed on disk since. |
| `GET /local/repos/watch` | SSE `change` events `{repo, kind: capnp\|dbc, file, path, deleted}`, from polling mtimes once a second. |
| `GET /local/device` / `POST /local/device` | `{host, root}`. POST takes `{host, key?}` and sets the device until restart. The host is `[user@]host` and the user defaults to `comma`. It's validated so it can never be an ssh option. |
| `GET /local/device/routes` | `{dongle, host, routes: [{route, log_id, segments: {n: {file: size}}, mtime, active: [n]}]}`, newest first. It comes from one `find` over ssh. |
| `POST /local/device/pull` | Body `{route (or a bare log_id), segs?, kinds? (default rlog, qlog)}`. Starts a `pull` job that copies with `ssh cat` into `ycpe/device/...` in the shared cache; the `device` source serves from there. Segments still recording (with a `.lock`) are skipped. |

The daemon never runs anything on a device except `cat` and `find`.

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
- **Namespaces.** device, internal, api and openpilotci copies of a file have the same bytes, so a copy from any one of them
  serves the others. commaCarSegments rlogs are filtered copies with the full rlog's name, so they're keyed by their HF URL
  only. They're served only for `carSegments` requests, and in `auto` only when every other source missed.
- **Writes.** Downloads land as `sha256(url)`. Logs are also mirrored as url_file chunks plus the manifest, so `LogReader`
  then reads them with no network.
- **Budget.** The whole directory is LRU-pruned on mtime down to `--cache-budget` (20 GB by default). Temporary files and
  anything touched in the last 2 minutes are skipped.
- **Staleness.** comma's servers sometimes recompress a file in place under the same URL. At most once a day per file, a hit
  is checked in the background with a HEAD to its canonical URL. The checked-at time is kept in the `user.ycpe.checked`
  xattr, or in memory where xattrs aren't supported. On a size mismatch, the copy and its chunks are refetched and atomically
  replaced. The hit is still served at once, and carries `X-Ycpe-Cache: stale-refetch` instead of `hit` until the new copy is
  in place. Plain hits are `X-Ycpe-Cache: hit`, and misses are `miss`.
- **Daemon-made files.** Device pulls and other modules' outputs go under `ycpe/`:
  - `cache.derived(name)` returns a path for a module's output
  - `cache.fetch(dongle, log_id, seg, file, sources=None, job=None)` returns a local path, downloading into the cache if
    needed
