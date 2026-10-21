# ycpe local daemon

ycpe is a static web app, and works as one. Run it through `serve.py` with openpilot's Python and you also get a
small local server that does what a browser can't on its own.

## Using it

```bash
.venv/bin/python openpilot/tools/ycpe/serve.py          # open http://localhost:3000/
```

- **Port.** Any port works. athena (device RPC from the browser) only allows port 3000 itself, so on other ports the
  daemon relays those calls (the `athena` module).
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
| `--webrtcd-port N` | 5001 | The port of this PC's webrtcd, for live data at `/live/local`. |

What the daemon unlocks in the UI:

| Module | What you get |
|---|---|
| `auth` | Your `~/.comma/auth.json` sign-in (from `tools/lib/auth.py`). The app opens signed in, and the account menu says so. |
| `sources` | Logs from every source openpilot's LogReader uses: your disk, a pulled device, comma's internal storage, the comma API and openpilotci. That includes routes the browser can't reach. In the Files pane, the source picker enables these. |
| `cache` | One disk cache shared with cabana, jotpluggler, replay and LogReader, so a route downloaded by any of them loads instantly in all of them. The badge panel can cache a whole route ahead of time (logs or video) and evict it. Progress shows in the loading pill. |
| `repos` | The local cereal schema (for a "local checkout" schema), and opendbc DBCs. The CAN view can save DBCs straight back into the checkout's opendbc_repo. |
| `device` | Lists the routes on your comma device and copies them over ssh into the cache. Nothing on the device is ever changed. |
| `athena` | Relays the page's device calls (upload queue, upload requests, live streams) to athena on ports athena doesn't allow. |
| `billing` | Relays the page's referrals read to billing, which allows the same few origins as athena. |
| `webrtc` | Live data from a webrtcd on this PC (openpilot, a sim, replay) at `/live/local`: it passes the page's offer on. |
| `jobs` | Progress and cancel for anything long, shown in the badge panel. |
| `transcode` | H.264 copies of the HEVC cameras for browsers that can't decode HEVC, encoded on the GPU when there is one. |

Without the daemon these features are disabled, with a short note saying they need it.

**Is it safe?**
1. The daemon only listens on 127.0.0.1, and it refuses requests addressed to any other host name, which blocks DNS rebinding.
   On Linux it also refuses connections from other users on the machine. Elsewhere, treat it as single-user: any local
   user could load the page and use your sign-in.
2. Every API call needs a random token that only the page `serve.py` served carries. Other websites can't get it, because
   cross-origin requests get no CORS headers and are rejected, and no other site may frame the app.
3. It reads files only from the cache, your data dirs and your checkouts, and it writes only to the cache and to opendbc's DBC folder.
4. On its own it never modifies anything on a comma device or in your comma account. The only commands it runs on a device
   are `cat` and `find`. The athena relay passes on the page's own upload calls, with the page's sign-in.
5. Your JWT and the session token are never logged.

## For developers

```bash
PYTHONPATH=$PWD .venv/bin/python -m unittest discover -s openpilot/tools/ycpe/daemon/tests -t .   # device tests need ssh to localhost
```

The page finds the daemon through `<meta name="ycpe-daemon">` in index.html, plus `GET /local/caps`.

## Security

- It binds 127.0.0.1 only. On Linux, a connection whose client socket belongs to another uid (looked up once per
  connection with sock_diag) gets 403 on everything, static files included, since index.html carries the token.
- Every request, static files included, must carry `Host: localhost:<port>` or `127.0.0.1:<port>`. This
  stops DNS rebinding.
- `/local/*` rejects any `Origin` other than the daemon's own, and any `Sec-Fetch-Site` other than
  `same-origin` or `none`. It never sends CORS headers.
- Every `/local/*` route except `caps` needs the per-process session token. It goes in the `X-Ycpe-Token`
  header, or in `?tok=` for URLs the page can't set headers on (media elements, worker fetches). serve.py
  puts the token in index.html, and the request log never prints query strings.
- File reads go through `confine(path, roots)`, which resolves symlinks and rejects `..`.
- The athena relay is POST only, to `https://athena.comma.ai/<dongle id>`, for the few JSON-RPC methods the app calls.
  It forwards the page's `Authorization` and nothing else, and limits body and answer sizes.
- The billing relay is one GET, `https://billing.comma.ai/v1/referrals`, the same way. Prime's checkout, portal and
  cancel aren't relayed.

## Writing a module

Add `daemon/<name>.py` and list it in `server.MODULES`, which `/local/caps` lists for the page's `daemon.has(name)`. A
module that fails to import stops the daemon.

```python
from openpilot.tools.ycpe.daemon import File, HTTPError, Stream, config, confine, route, sse
from openpilot.tools.ycpe.daemon import jobs

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
| `GET /local/jobs/events` | SSE `event: job`, the current jobs on connect and then every update: `{id, kind, title, state: running\|done\|error\|cancelled, done, total, unit, text, result, error, started, ended, meta}` |
| `POST /local/jobs/{id}/cancel` | asks the job to stop |

On the page, `daemon.js` exports `daemon.{present, caps, has(module), fetch(path, opts), json(path, opts),
url(path), events(path, fn)}` and `needsDaemon(what)`.

## Endpoints

| endpoint | |
|---|---|
| `GET /local/caps` | `{daemon, version, modules: [name], sources: [name], commit: {sha, time}}`. This is the only route that needs no token. |
| `GET /local/auth` | `{token}`: the JWT from `~/.comma/auth.json`, or null. |
| `POST /local/sources/resolve` | Body `{route, segs?, kinds?, sources?, sizes?}`. Returns `{route, segments: {n: {kind: {url, file, size, source, cached}}}, missing: {n: [kind]}, errors: {source: msg}}`. Details below. |
| `GET /local/file/{source}/{dongle}/{log_id}/{seg}/{file}` | The bytes, with Range support. It serves from the shared cache. On a miss it streams while the download fills the cache. A Range request on a miss is proxied, and the full download starts in the background. |
| `GET /local/cache[?route=&segs=<count>]` | `{dir, budget, bytes, files}`. With `route`, it adds per-segment `{kind: {file, source, size}}` of what's cached. |
| `POST /local/cache/prefetch` | Body as resolve. Starts a `prefetch` job (8 downloads at once). `meta.segs` gives each segment's state: queued, loading, disk or error. |
| `POST /local/cache/evict` | Body `{route, segs?, kinds?}`. Deletes the local cached copies. |
| `GET /local/repos` | `[{id, name, git: {branch, commit, dirty}, cereal, opendbc: {dbc}}]`. These are the `--repo` checkouts (the running one by default). |
| `GET /local/repos/{id}/capnp` | `{files: {import path: text}}`: cereal's `*.capnp` (including `include/c++.capnp`) and opendbc's `car.capnp`. |
| `GET /local/repos/{id}/dbc` | `[{name}]` |
| `GET /local/repos/{id}/dbc/{name}` | `{name, text, mtime}` |
| `PUT /local/repos/{id}/dbc/{name}[?mtime=]` | Body is the DBC text. Saves it atomically into opendbc's dbc directory, and only there: the name must match `[A-Za-z0-9_.-]+\.dbc`, and the path is confined. With the `mtime` from the read, it answers 409 if the file changed on disk since. |
| `GET /local/repos/watch` | SSE `change` events `{repo, kind: capnp, file, path, deleted}` when cereal's `.capnp` files change, from polling their mtimes once a second. |
| `GET /local/device` / `POST /local/device` | `{host, root}`. POST takes `{host}` and sets the device until restart. The host is `[user@]host` and the user defaults to `comma`. It's validated so it can never be an ssh option. |
| `GET /local/device/routes` | `{dongle, host, routes: [{route, log_id, segments: {n: {file: size}}, mtime, active: [n]}]}`, newest first. It comes from one `find` over ssh. |
| `POST /local/webrtc/stream` | Body: webrtcd's stream request (`{sdp, cameras, enabled, ...}`). Forwards it to `127.0.0.1:<--webrtcd-port>/stream`, and nowhere else, and returns webrtcd's answer. |
| `POST /local/device/pull` | Body `{route (or a bare log_id), segs?, kinds? (default rlog, qlog)}`. Starts a `pull` job that copies with `ssh cat` into `ycpe/device/...` in the shared cache; the `device` source serves from there. Segments still recording (with a `.lock`) are skipped. |

The daemon never runs anything on a device except `cat` and `find`.

Resolve parameters:
- `route` is `dongle|log_id`, and `segs` a list of segment numbers. The page names them from the route's record; without
  `segs` (here and in the cache endpoints) they're the segments on this machine's disk: local log directories and device pulls.
- `kinds` is a list of `rlog qlog qcamera fcamera ecamera dcamera log`. `log` means the rlog, else the qlog, for each segment.
- `sources` is `"auto"` or a list. The auto order is device → local → internal → api → openpilotci. A copy already in the shared cache wins over any download.
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
- **Sources.** device, internal, api and openpilotci copies of a file have the same bytes, so a copy from any one of them
  serves the others.
- **Writes.** Downloads land as `sha256(url)`. Logs are also mirrored as url_file chunks plus the manifest, so `LogReader`
  then reads them with no network.
- **Budget.** The whole directory is LRU-pruned on mtime down to `--cache-budget` (20 GB by default). Temporary files and
  anything touched in the last 2 minutes are skipped.
- **Staleness.** comma's servers sometimes recompress a file in place under the same URL. At most once a day per file, a hit
  is checked in the background with a HEAD to its canonical URL. The checked-at time is kept in the `user.ycpe.checked`
  xattr, or in memory where xattrs aren't supported. On a size mismatch, the copy and its chunks are refetched and atomically
  replaced; the old copy is still served until the new one is in place. Hits carry `X-Ycpe-Cache: hit`, and misses
  `miss`.
- **Daemon-made files.** Device pulls and transcodes go under `ycpe/`, one directory per module (`cache.ycpe_dir(name)`).
