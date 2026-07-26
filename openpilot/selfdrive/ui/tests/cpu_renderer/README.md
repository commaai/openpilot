# MICI CPU renderer prototype

This directory contains a small standalone benchmark for the native raster core
used by the GPU-free MICI UI backend. The production integration lives in
`openpilot/system/ui/lib/cpu_backend.py`. It is selected automatically on
MICI; use `RAYLIB_BACKEND=cpu` for PC testing or `RAYLIB_BACKEND=comma` as a
device fallback.

Run:

```bash
python openpilot/selfdrive/ui/tests/cpu_renderer/prototype.py \
  --output /tmp/mici_cpu_renderer.ppm
```

The prototype builds `openpilot/system/ui/lib/cpu_renderer/renderer.c` and
renders a representative 536x240 overlay into a premultiplied BGRA
framebuffer. The integrated backend additionally provides the raylib-compatible
shape, texture, text, camera, input, render-target, shader-effect, and lifecycle
surface used by the UI. On MICI it presents rotated double-buffered DRM dumb
buffers and retains the `/tmp/drmfd.sock` control connection for the complete
window lifetime.

For representative end-to-end numbers, run
`openpilot/selfdrive/ui/tests/profile_onroad.py` with `CPU_RENDER_PROFILE=1`.
On a production-configured MICI, the harness pins the UI loop to core 5 just
like `ui.py`; asynchronous camera conversion drops realtime scheduling and is
pinned to core 4.
