"""Raylib facade used by comma's UIs.

The default path is intentionally a transparent proxy to pyray. MICI selects
the software implementation automatically; ``RAYLIB_BACKEND=cpu`` also enables
it explicitly on development hosts, while an explicit non-CPU backend remains
an escape hatch. Keeping the selection here preserves the existing
raylib-shaped UI API while allowing MICI to render and present without opening
the GPU.
"""

import os
import platform
from enum import EnumType


def _is_mici() -> bool:
  if not os.path.isfile("/TICI"):
    return False
  try:
    with open("/sys/firmware/devicetree/base/model") as model_file:
      return model_file.read().strip("\x00").split("comma ")[-1] == "mici"
  except OSError:
    return False


_requested_backend = os.getenv("RAYLIB_BACKEND")
CPU_BACKEND = (_requested_backend or "").lower() == "cpu" or (_requested_backend is None and _is_mici())

# comma-deps-raylib does not know about the in-tree CPU backend.
# Its headless module provides the same CFFI ABI types on current development
# hosts. The deployed MICI dependency predates that module, so select its
# desktop ABI there; unlike the comma ABI, it does not load Adreno EGL/GLES.
# Restore the requested backend after importing it.
if CPU_BACKEND:
  os.environ["RAYLIB_BACKEND"] = "desktop" if platform.machine() in ("aarch64", "arm64") else "headless"
import pyray as _pyray
if CPU_BACKEND:
  if _requested_backend is None:
    os.environ.pop("RAYLIB_BACKEND", None)
  else:
    os.environ["RAYLIB_BACKEND"] = _requested_backend

# cffi is used directly by a few UI call sites to allocate shader uniform
# values. It remains ABI-compatible in both modes.
ffi = _pyray.ffi

_CPU_SAFE_PYRAY = {
  "Color", "Font", "GlyphInfo", "Image", "Matrix", "Rectangle",
  "RenderTexture", "Shader", "Texture", "Vector2", "Vector3", "Vector4",
  "check_collision_point_rec", "check_collision_recs", "color_from_hsv",
  "color_to_hsv", "color_to_int", "get_collision_rec", "matrix_ortho",
}


def __getattr__(name: str):
  # Functions implemented by the CPU backend are installed here as explicit
  # module globals. Reject missing rendering entry points instead of silently
  # falling through to a GPU-backed implementation.
  value = getattr(_pyray, name)
  if CPU_BACKEND and callable(value) and name not in _CPU_SAFE_PYRAY and not isinstance(value, EnumType):
    raise NotImplementedError(f"raylib CPU backend does not implement {name}")
  return value


def using_cpu_backend() -> bool:
  return CPU_BACKEND


if CPU_BACKEND:
  from openpilot.system.ui.lib.cpu_backend import EXPORTED as _CPU_EXPORTED
  globals().update(_CPU_EXPORTED)
