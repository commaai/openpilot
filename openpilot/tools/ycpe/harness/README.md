# YCPE harness

```bash
./run.py                 # every scenario except slow ones, ~40 s; out/latest/report.{html,md}, previous run in out/prev
./run.py -k plot -k can  # scenarios whose name contains "plot" or "can" (slow ones included)
./run.py --loop 5        # run 5 times (--loop 0: until ^C), each diffed against the one before
./run.py --trace         # also a Chrome trace per scenario: out/latest/traces/<scenario>.json, opens in ui.perfetto.dev
./trace.py out/latest/traces/layout-drag.json  # the longest frame's events by self time, and the panes it laid out and painted
./run.py --static        # serve.py without the local daemon (default: with it, through the openpilot venv)
./run.py --slow          # include scenarios marked "slow" (e.g. plot-repeat10)
./run.py --bless         # also store this run's shots as goldens (harness/golden/); later runs diff against them
./report.py              # rebuild the report from out/latest/results.json
./reference.py [names]   # out/reference/*.png: connect.comma.ai, native jotpluggler and cabana (see --help)
```

The scripts run through `uv run --script`, which provides playwright, pillow and numpy; Chrome is `/usr/bin/google-chrome`.

## Scenarios

`scenarios/*.json`:

```js
{name, url, viewport?: {width, height, mobile?}, auth?: true /* ~/.comma/auth.json JWT as comma_jwt */,
 cpu_throttle?: 4, timeout?: ms, slow?: true, allow?: [regex for expected errors],
 lint_ignore?: [css selectors], budgets?: {loaf_ms: 50, work_p99_ms: 8, drops: null},
 reference?: [reference names], routes?: [{url: regex, method?, rpc?, json | file | body, status?}], steps: [...]}
```

`routes` answer matching requests (the page's and its workers') with canned JSON, a file under `harness/`
(`fixtures/`), or text, so a scenario needn't depend on what a real device has. `method` limits one to that HTTP
method and `rpc` to that athena method, so a mutating call can be faked while reads stay real. The page's mutating
fetches, answered or blocked, are in `window.__sent` as `{method, url, body}` for asserts on exactly what was sent.

Steps: `{click: sel | [x, y]}`, `{hover: sel | [x, y]}`, `{drag: {from, to | by: [dx, dy], steps}}`,
`{dragTo: [sel, sel]}` (HTML drag and drop), `{wheel: {at?, dx?, dy, every?: ms}, repeat?}`, `{fill: [sel, text]}`, `{viewport: [w, h]}`, `{key: "Shift+ArrowRight", repeat?}`, `{select: [sel, value]}`, `{wait: ms}`,
`{waitFor: sel | {fn: js}}`, `{goto: path}`, `{eval: js}`, `{note: js}` (its value goes in the report), `{assert: js}`, `{shot: name}`,
`{network: {down_kbps, up_kbps?, latency_ms?} | "online"}` (throttles the page and its workers),
`{filmstrip: {name, every: ms, until: js, crop?: [sel]}}` (a screenshot every `every` ms until `until` holds, composed
into `shots/<scenario>/<name>.strip.png` from the `crop` boxes). Any step takes `timeout`. An assert may return a
promise; returning a string fails it with that string as the reason. An assert with `rule: "drops"` (or another perf rule)
fails as that rule, so it only warns when the machine is too busy to measure.

## What fails a scenario

- a mutating request (POST/PATCH/PUT/DELETE) to the comma API or athena (directly or through the daemon's relay); it is blocked and recorded, and `allow`
  can't hide it. Athena's read-only RPCs (`ATHENA_READS` in run.py) pass.
- a page error, a console error, a failed step or assert
- a long animation frame over `loaf_ms`, perf.js work p99 over `work_p99_ms`, dropped frames over `drops`, a first
  contentful paint later than `first_paint_ms` (150). Frames before the first paint are page load: they are listed
  as info and left out of the frame stats.

Scenarios run signed out unless they set `auth`, even though the daemon offers `~/.comma/auth.json`'s JWT.

Time spent taking screenshots and linting is excluded from the frame stats (shaded in the charts). When an empty page
drops frames or the load average exceeds the cpu count, the perf budgets only warn. A scenario that hits a network flake
(ERR_NETWORK_CHANGED: USB ethernet devices on this box come and go) is retried once.

## Findings

The report groups failures, events and lint issues across scenarios into findings by rule and selector, each with the
file it points at (a local script in a long frame or error stack, else the module that creates the innermost class in
its selector). `out/latest/findings.json` keeps them, and every run lists what is new, fixed and still open since the
previous one, grouped by file. Only scenarios that ran can fix a finding. Long frames say which step they happened during.

Layout lint findings (overflow, sideways scroll, clipped text, overlap, obscured controls, small targets, layout shift,
broken images) are reported but don't fail the scenario. A native-select finding (a `<select>` some rule took off the in-page, themed picker, see
style.css) is an error.

## Modes (`--mode`)

- `gpu` (default): headless Chrome on the NVIDIA GPU through ANGLE/EGL. Hardware raster, compositing and WebGL,
  60 Hz begin-frames from a timer. Main-thread and GPU costs are real; there is no display, so no vsync jitter or
  present latency.
- `soft`: headless SwiftShader. Main-thread numbers hold, raster and WebGL costs are inflated. For machines without a GPU.
- `xvfb`: a headful window on a private Xvfb (llvmpipe). The real window and input path, with software GL.

Frame numbers are only meaningful when the machine is not saturated; the report header shows the load average.
