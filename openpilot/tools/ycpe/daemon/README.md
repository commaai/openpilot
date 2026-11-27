# ycpe local daemon

ycpe is a static web app, and works as one. Run it through `serve.py` with openpilot's Python and you also get a
small local server that does what a browser can't on its own. A **local** badge in the top bar means it's on.

```bash
.venv/bin/python openpilot/tools/ycpe/serve.py          # http://localhost:3000/; --static for the plain app
```

| Option | Default | |
|---|---|---|
| `--data-dir DIR` | | A folder of route segments (`<log id>--<n>/rlog.zst`, as on a device). It can be repeated. |
| `--webrtcd-port N` | 5001 | This PC's webrtcd, for live data at `/live/local`. |

| Module | What you get |
|---|---|
| `auth` | Your `~/.comma/auth.json` sign-in: the app opens signed in. |
| `sources` | Logs from every source LogReader uses: your disk, comma's internal storage, the comma API and openpilotci. |
| `cache` | One disk cache shared with LogReader and FrameReader, so a route either of them downloaded opens instantly. |
| `webrtc` | Live data from a webrtcd on this PC (openpilot, a sim, replay). |

## Security

- It binds 127.0.0.1. Every request must be addressed to one of its own host names, which
  stops DNS rebinding. On Linux, connections from another user on the machine get 403 on everything.
- Every `/local/*` route except `caps` needs the per-process session token (`X-Ycpe-Token`, or `?tok=` for media and
  workers), which only the page `serve.py` served carries. It rejects foreign `Origin` and cross-site `Sec-Fetch-Site`,
  and never sends CORS headers.
- It reads only the cache and your data dirs, and writes only the cache; paths go through `confine()`.
- It never changes anything in your comma account: its only call to the API is a route's file listing.
- Your JWT and the session token are never logged.

## Shared cache

`Paths.download_cache_root()`, in `url_file.py`'s 1 MB chunks, the cache LogReader and FrameReader already use, held under
its size by url_file's LRU. A file is found under the canonical URL of every source that could hold it.

## Developing

Each module is `daemon/<name>.py`, listed in `server.MODULES`, with handlers declared by `@route(method, pattern)` (see `core.py`).

```bash
PYTHONPATH=$PWD .venv/bin/python -m unittest discover -s openpilot/tools/ycpe/daemon/tests -t .
```
