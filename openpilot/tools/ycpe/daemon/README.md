# ycpe local daemon

ycpe is a static web app, and works as one. Run it through `serve.py` with openpilot's Python and you also get a
small local server that does what a browser can't on its own. A **local** badge in the top bar means it's on.

```bash
.venv/bin/python openpilot/tools/ycpe/serve.py          # http://localhost:3000/; --static for the plain app
```

| Option | Default | |
|---|---|---|
| `--cache-budget GB` | 20 | Size of the shared download cache, pruned least-recently-used first. |
| `--device HOST` | `$YCPE_DEVICE` | A comma device on your network, `[user@]host` (user `comma` by default). |
| `--device-key FILE` | `$YCPE_DEVICE_KEY` | The ssh key for the device. |
| `--data-dir DIR` | | A folder of route segments (`<log id>--<n>/rlog.zst`, as on a device). It can be repeated. |
| `--repo DIR` | the checkout `serve.py` is in | An openpilot checkout whose opendbc the CAN view saves DBCs into. |
| `--webrtcd-port N` | 5001 | This PC's webrtcd, for live data at `/live/local`. |
| `--host IP` | | Also serve on this LAN address. Every visitor gets your comma sign-in. |

| Module | What you get |
|---|---|
| `auth` | Your `~/.comma/auth.json` sign-in: the app opens signed in. |
| `sources` | Logs from every source LogReader uses: your disk, a pulled device, comma's internal storage, the comma API and openpilotci. |
| `cache` | One disk cache shared with cabana, jotpluggler, replay and LogReader, so a route any of them downloaded opens instantly. The badge panel caches or evicts a whole route. |
| `repos` | Saving DBCs from the CAN view straight into the checkout's `opendbc_repo`. |
| `device` | Lists the routes on your comma device and copies them over ssh into the cache. |
| `athena`, `billing` | Relays the page's device calls and referrals read on origins athena and billing don't allow. |
| `webrtc` | Live data from a webrtcd on this PC (openpilot, a sim, replay). |
| `transcode` | H.264 copies of the HEVC cameras for browsers that can't decode HEVC. |
| `jobs` | Progress and cancel for anything long, in the badge panel. |

## Security

- It binds 127.0.0.1 (all interfaces with `--host`). Every request must be addressed to one of its own host names, which
  stops DNS rebinding. On Linux, connections from another user on the machine get 403 on everything.
- Every `/local/*` route except `caps` needs the per-process session token (`X-Ycpe-Token`, or `?tok=` for media and
  workers), which only the page `serve.py` served carries. It rejects foreign `Origin` and cross-site `Sec-Fetch-Site`,
  and never sends CORS headers.
- It reads only the cache, your data dirs and checkouts, and writes only the cache and opendbc's DBC folder; paths go
  through `confine()`.
- It never changes anything on a device or in your comma account: on a device it runs only `cat` and `find`, and the
  relays pass on just the page's own calls, with the page's sign-in.
- Your JWT and the session token are never logged.

## Shared cache

`Paths.download_cache_root()`, the directory cabana, jotpluggler, replay (`file_downloader.py`) and LogReader
(`url_file.py`) already use. A file is found under the canonical URL of every source that could hold it, as a whole file,
a decompressed one or complete url_file chunks; downloads land whole and logs are mirrored as chunks too. It's
LRU-pruned to the budget, and a hit is checked against its source at most once a day, refetched if the size changed.
Device pulls and transcodes live under `ycpe/`.

## Developing

Each module is `daemon/<name>.py`, listed in `server.MODULES` (the page asks `daemon.has(name)`), with handlers declared by
`@route(method, pattern)` (see `core.py`). Long work runs as a job (`jobs.py`).

```bash
PYTHONPATH=$PWD .venv/bin/python -m unittest discover -s openpilot/tools/ycpe/daemon/tests -t .   # device tests need ssh to localhost
```
