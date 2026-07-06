# Loggy

Loggy is a GLFW/OpenGL route/browser/replay viewer. Build artifacts live in this
directory (`_loggy`, launcher wrappers, and smoke test binaries).

## Build

```bash
scons -j$(nproc) openpilot/tools/loggy/_loggy \
  openpilot/tools/loggy/loggy \
  openpilot/tools/loggy/loggy_cabana \
  openpilot/tools/loggy/loggy_jotpluggler
```

## Run

```bash
openpilot/tools/loggy/loggy --demo
openpilot/tools/loggy/loggy_cabana --demo
openpilot/tools/loggy/loggy_jotpluggler --demo
openpilot/tools/loggy/loggy_cabana --device 192.168.0.10
openpilot/tools/loggy/loggy_cabana --socketcan vcan0
```

## Tests

```bash
bash openpilot/tools/loggy/tests/run_smoke.sh
```

The default smoke run builds the `loggy_smoke_build` SCons alias and executes
the deterministic non-GUI smoke binaries. Add `--with-route` for the real-route
ingest smoke, `--with-capture` for virtual-display preset screenshots, or
`--full` for both.

## Headless Capture Smoke (CI friendly)

Run both preset capture checks under a virtual display:

```bash
bash openpilot/tools/loggy/tests/capture_presets.sh
```

The script exercises:
- `loggy_cabana --demo --output ...`
- `loggy_jotpluggler --demo --output ...`

Under the hood it prefers the repo venv `Xvfb` wrapper used by the GUI harness
and falls back to `xvfb-run`. The default capture size is 1280x720. If your CI
runner requires fixed output names, set `LOGGY_CAPTURE_DIR`.
