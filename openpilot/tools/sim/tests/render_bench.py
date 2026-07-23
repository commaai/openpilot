#!/usr/bin/env python3
"""Standalone MetaDrive render-throughput benchmark for CI tuning (issue #30693).

Runs each render-config variant in a fresh subprocess on the same machine and
reports steady-state render fps, isolating renderer cost from the rest of the
openpilot stack and from runner-to-runner CPU variance.

Self-contained on purpose: only needs metadrive + panda3d + numpy (no scons build).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

W, H = 1928, 1208

VARIANTS = {
  "simple-noshadow-half":      {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                "METADRIVE_NO_SHADOWS": "1"},
  "simple-noshadow-noterrain-half": {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                     "METADRIVE_NO_SHADOWS": "1", "METADRIVE_NO_TERRAIN": "1"},
  "simple-noshadow-cheapterrain-half": {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                        "METADRIVE_NO_SHADOWS": "1", "METADRIVE_CHEAP_TERRAIN": "1"},
  "flatcard-half":             {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1", "METADRIVE_RENDER_SCALE": "0.5",
                                "METADRIVE_NO_SHADOWS": "1", "METADRIVE_FLAT_TERRAIN_CARD": "1"},
  "flatcard-full":             {"METADRIVE_SIMPLE_RENDER": "1", "METADRIVE_NO_MSAA": "1",
                                "METADRIVE_NO_SHADOWS": "1", "METADRIVE_FLAT_TERRAIN_CARD": "1"},
}

PYSPY_VARIANTS = ()

OUT_DIR = "/tmp/render_bench"


def save_ppm(path, img):
  with open(path, "wb") as f:
    f.write(b"P6\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
    f.write(np.ascontiguousarray(img).tobytes())

WARMUP_FRAMES = 30
BENCH_FRAMES = 100


def measure():
  # env flags must be applied before the engine is created
  from openpilot.tools.sim.bridge.metadrive.ci_render_patches import apply_ci_render_patches
  apply_ci_render_patches()

  from metadrive.component.map.pg_map import MapGenerateMethod
  from metadrive.envs.metadrive_env import MetaDriveEnv
  from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad

  scale = float(os.environ.get("METADRIVE_RENDER_SCALE", "1"))
  rw, rh = round(W * scale), round(H * scale)

  def straight(length):
    return {"id": "S", "pre_block_socket_index": 0, "length": length}

  def curve(length, angle=45, direction=0):
    return {"id": "C", "pre_block_socket_index": 0, "length": length, "radius": length, "angle": angle, "dir": direction}

  ts = 60
  config = {
    "use_render": False,
    "vehicle_config": {"enable_reverse": False, "render_vehicle": False, "image_source": "rgb_road"},
    "sensors": {"rgb_road": (RGBCameraRoad, rw, rh)},
    "image_on_cuda": False,
    "image_observation": True,
    "interface_panel": [],
    "out_of_route_done": False,
    "on_continuous_line_done": False,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "traffic_density": 0.0,
    "map_config": {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2, "lane_width": 4.5,
                    "config": [None, straight(ts), curve(ts * 2, 90), straight(ts), curve(ts * 2, 90),
                            straight(ts), curve(ts * 2, 90), straight(ts), curve(ts * 2, 90)]},
    "decision_repeat": 1,
    "physics_world_step_size": 0.05,
    "preload_models": False,
    "show_logo": False,
    "anisotropic_filtering": False,
    "show_terrain": not bool(os.environ.get("METADRIVE_NO_TERRAIN")),
  }

  env = MetaDriveEnv(config)
  env.reset()
  cam = env.engine.sensors["rgb_road"]
  cam.get_cam().reparentTo(env.agent.origin)

  def frame():
    env.step([0, 0.2])
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get()
    if img.shape[0] != H or img.shape[1] != W:
      img = img.repeat(H // img.shape[0], axis=0).repeat(W // img.shape[1], axis=1)
    return img

  for _ in range(WARMUP_FRAMES):
    img = frame()
  name = os.environ.get("RENDER_BENCH_NAME", "variant")
  os.makedirs(OUT_DIR, exist_ok=True)
  save_ppm(os.path.join(OUT_DIR, f"{name}.ppm"), img)
  t0 = time.monotonic()
  for _ in range(BENCH_FRAMES):
    frame()
  dt = time.monotonic() - t0
  print(json.dumps({"fps": round(BENCH_FRAMES / dt, 2)}))
  env.close()


if __name__ == "__main__":
  if "--measure" in sys.argv:
    measure()
    sys.exit(0)

  results = {}
  for name, flags in VARIANTS.items():
    env = os.environ.copy()
    env.update(flags)
    env["RENDER_BENCH_NAME"] = name
    try:
      child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--measure"],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      prof = None
      if os.environ.get("RENDER_BENCH_PYSPY") and name in PYSPY_VARIANTS:
        os.makedirs(OUT_DIR, exist_ok=True)
        prof = subprocess.Popen(["sudo", "-n", "env", "PATH=" + os.environ["PATH"],
                                 "py-spy", "record", "--native", "-d", "45", "-r", "50", "-f", "speedscope",
                                 "-o", os.path.join(OUT_DIR, f"pyspy_{name}.json"), "--pid", str(child.pid)])
      stdout, stderr = child.communicate(timeout=600)
      if prof is not None:
        prof.wait()
      line = [l for l in stdout.splitlines() if l.startswith("{")]
      if line:
        results[name] = json.loads(line[-1])["fps"]
      else:
        err = [l for l in stderr.splitlines() if l.strip()]
        results[name] = f"failed (rc={child.returncode}): {' | '.join(err[-5:])[:500]}"
    except Exception as e:
      results[name] = f"error: {e}"
    print(f"{name:35s} {results[name]}", flush=True)

  print("\n=== RENDER BENCH RESULTS (target: >=20 fps) ===")
  for name, fps in results.items():
    print(f"{name:35s} {fps}")
