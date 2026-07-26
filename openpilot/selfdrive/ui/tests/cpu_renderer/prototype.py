#!/usr/bin/env python3
"""Build and benchmark the minimal MICI CPU rasterizer prototype."""

import argparse
import ctypes
import pathlib
import statistics
import subprocess
import tempfile
import time

import numpy as np

WIDTH = 536
HEIGHT = 240
HERE = pathlib.Path(__file__).resolve().parents[4] / "system/ui/lib/cpu_renderer"


class Surface(ctypes.Structure):
  _fields_ = [
    ("pixels", ctypes.POINTER(ctypes.c_uint8)),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("stride", ctypes.c_int),
    ("clip_x0", ctypes.c_int),
    ("clip_y0", ctypes.c_int),
    ("clip_x1", ctypes.c_int),
    ("clip_y1", ctypes.c_int),
  ]


def build() -> tuple[ctypes.CDLL, tempfile.TemporaryDirectory]:
  tmp = tempfile.TemporaryDirectory(prefix="mici_cpu_renderer_")
  library = pathlib.Path(tmp.name) / "renderer.so"
  subprocess.run([
    "cc", "-O3", "-march=native", "-shared", "-fPIC", "-I/usr/include/libdrm",
    str(HERE / "renderer.c"), "-lm", "-ldrm", "-o", str(library),
  ], check=True)
  lib = ctypes.CDLL(str(library))
  lib.sr_clear.argtypes = [ctypes.POINTER(Surface), ctypes.c_uint32]
  lib.sr_rect.argtypes = [
    ctypes.POINTER(Surface), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
  ]
  lib.sr_gradient_v.argtypes = [
    ctypes.POINTER(Surface), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint32, ctypes.c_uint32,
  ]
  lib.sr_circle.argtypes = [
    ctypes.POINTER(Surface), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
  ]
  lib.sr_demo_frame.argtypes = [ctypes.POINTER(Surface), ctypes.c_int]
  return lib, tmp


def save_ppm(path: pathlib.Path, framebuffer: np.ndarray) -> None:
  # The framebuffer is BGRA; PPM expects RGB.
  rgb = framebuffer[:, :, 2::-1]
  with path.open("wb") as f:
    f.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode())
    f.write(rgb.tobytes())


def draw_scene(lib: ctypes.CDLL, surface: Surface, frame: int) -> None:
  """Exercise the backend as individual raylib-shaped draw calls."""
  s = ctypes.byref(surface)
  lib.sr_clear(s, 0xff101820)
  lib.sr_gradient_v(s, 0, 120, WIDTH, 120, 0x00101820, 0xd0000000)
  pulse = frame % 20
  lib.sr_circle(s, 472, 34, 18 + pulse // 8, 0xe020c060)
  lib.sr_rect(s, 12, 12, 92, 54, 0xd0181818)
  lib.sr_rect(s, 20, 22, 55 + pulse, 8, 0xfff0f0f0)
  lib.sr_rect(s, 20, 40, 34, 8, 0xffa0a0a0)


def benchmark(draw, frames: int) -> tuple[float, list[float]]:
  samples = []
  start = time.perf_counter()
  for frame in range(frames):
    frame_start = time.perf_counter_ns()
    draw(frame)
    samples.append((time.perf_counter_ns() - frame_start) / 1e6)
  return time.perf_counter() - start, samples


def print_results(name: str, elapsed: float, samples: list[float]) -> None:
  samples.sort()
  print(f"{name}: throughput={len(samples) / elapsed:.1f} fps")
  result = f"frame_ms mean={statistics.fmean(samples):.3f} p50={samples[len(samples) // 2]:.3f} "
  result += f"p95={samples[int(len(samples) * .95)]:.3f} p99={samples[int(len(samples) * .99)]:.3f}"
  print(f"{name}: {result}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--frames", type=int, default=3000)
  parser.add_argument("--output", type=pathlib.Path)
  args = parser.parse_args()

  lib, build_dir = build()
  framebuffer = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
  surface = Surface(
    framebuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    WIDTH, HEIGHT, framebuffer.strides[0],
    0, 0, WIDTH, HEIGHT,
  )

  # Warm caches and dynamic linker state.
  for frame in range(100):
    lib.sr_demo_frame(ctypes.byref(surface), frame)

  print(f"surface={WIDTH}x{HEIGHT} frames={args.frames}")
  elapsed, samples = benchmark(lambda frame: lib.sr_demo_frame(ctypes.byref(surface), frame), args.frames)
  print_results("batched", elapsed, samples)

  # This intentionally measures the upper bound of crossing Python/C once per
  # primitive rather than submitting a native batch.
  elapsed, samples = benchmark(lambda frame: draw_scene(lib, surface, frame), args.frames)
  print_results("individual", elapsed, samples)

  if args.output:
    save_ppm(args.output, framebuffer)
    print(f"wrote {args.output}")

  # Keep the TemporaryDirectory alive until after all native calls.
  build_dir.cleanup()


if __name__ == "__main__":
  main()
