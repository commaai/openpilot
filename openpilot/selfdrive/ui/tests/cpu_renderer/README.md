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
renders a representative 536x240 overlay into a premultiplied RGBA
framebuffer. The integrated backend additionally provides the raylib-compatible
shape, texture, text, camera, input, render-target, shader-effect, and lifecycle
surface used by the UI. On MICI it renders directly into landscape
double-buffered cached MSM GEM buffers and uses SDM845's inline MDP rotator to scan
them out through the panel's native portrait DSI mode. It retains the
`/tmp/drmfd.sock` control connection for the complete window lifetime. The
MICI kernel must expose `SDE_PIX_FMT_RGBA_8888` and its tiled output in the
inline SBUF format tables; the stock table otherwise limits inline rotation to
YUV inputs. The renderer uses cached MSM GEM scanout buffers with explicit CPU
prepare/finalize synchronization so rasterization stays cacheable without a
full-frame copy. The default MICI path also imports road-camera NV12 DMA-BUFs
as a second atomic KMS
plane. The VIG rotator performs rotation and integral 2x downscale, then the
QSEED/CSC path finishes scaling, color conversion, and composition under the
transparent CPU overlay. Set `CPU_MDP_CAMERA=0` to force the CPU
camera-conversion fallback.

For representative end-to-end numbers, run
`openpilot/selfdrive/ui/tests/profile_onroad.py` with `CPU_RENDER_PROFILE=1`.
On a production-configured MICI, the harness pins the UI loop to core 5 just
like `ui.py`. Use `--camera-buffers 18 --camera-every 1` to exercise the full
camerad buffer-ring size. The asynchronous camera converter is used only as a
fallback (including the nonlinear driver-camera enhancement); it drops
realtime scheduling and is pinned to core 4.
