# YCPE harness

```bash
./run.py                 # every scenario except slow ones, ~40 s; out/latest/report.{html,md}, previous run in out/prev
./run.py -k plot         # scenarios whose name contains "plot" (slow ones included)
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
 reference?: [reference names], steps: [...]}
```

Steps: `{click: sel | [x, y]}`, `{hover: sel | [x, y]}`, `{drag: {from, to | by: [dx, dy], steps}}`,
`{wheel: {at?, dx?, dy}}`, `{key: "Shift+ArrowRight", repeat?}`, `{select: [sel, value]}`, `{wait: ms}`,
`{waitFor: sel | {fn: js}}`, `{goto: path}`, `{eval: js}`, `{assert: js}`, `{shot: name}`. Any step takes `timeout`.

## What fails a scenario

- a mutating request (POST/PATCH/PUT/DELETE) to the comma API or athena; it is blocked and recorded
- a page error, a console error, a failed step or assert
- a long animation frame over `loaf_ms`, perf.js work p99 over `work_p99_ms`, dropped frames over `drops`

Time spent taking screenshots and linting is excluded from the frame stats (shaded in the charts).
Layout lint findings (overflow, sideways scroll, clipped text, overlap, obscured controls, small targets, layout shift,
broken images)
are reported but don't fail the scenario.

## Modes (`--mode`)

- `gpu` (default): headless Chrome on the NVIDIA GPU through ANGLE/EGL. Hardware raster, compositing and WebGL,
  60 Hz begin-frames from a timer. Main-thread and GPU costs are real; there is no display, so no vsync jitter or
  present latency.
- `soft`: headless SwiftShader. Main-thread numbers hold, raster and WebGL costs are inflated. For machines without a GPU.
- `xvfb`: a headful window on a private Xvfb (llvmpipe). The real window and input path, with software GL.

Frame numbers are only meaningful when the machine is not saturated; the report header shows the load average.
