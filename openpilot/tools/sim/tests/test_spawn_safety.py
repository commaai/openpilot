import importlib
import unittest
from unittest import mock

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.lib.camerad import rgb_to_nv12, W, H
from openpilot.tools.sim.lib.common import SIM_MP_CTX, World

try:
  metadrive_process = importlib.import_module("openpilot.tools.sim.bridge.metadrive.metadrive_process")
except ModuleNotFoundError:
  metadrive_process = None


class FakeWorld(World):
  def apply_controls(self, steer_sim, throttle_out, brake_out, /):
    pass

  def tick(self):
    pass

  def read_state(self):
    pass

  def read_sensors(self, simulator_state, /):
    pass

  def read_cameras(self):
    pass

  def close(self, reason: str):
    pass

  def reset(self):
    pass


class FakeBridge(SimulatorBridge):
  def spawn_world(self, q, /) -> World:
    return FakeWorld(self.dual_camera)


def _report_state(bridge, world, q):
  world.exit_event.set()
  q.put({
    "cls": type(bridge).__name__,
    "dual_camera": bridge.dual_camera,
    "high_quality": bridge.high_quality,
    "started": bool(bridge.started.value),
    "has_params": bridge.params is not None,
    "world": bridge.world,
    "image_lock": world.image_lock.acquire(timeout=5),
  })


class TestSpawnSafety(OpenpilotTestCase):
  def test_spawn_start_method(self):
    # an OpenGL context can't survive fork(), so the sim never relies on the platform default
    assert SIM_MP_CTX.get_start_method() == "spawn"

  def test_shared_objects_use_sim_context(self):
    # mixing start methods for shared synchronization primitives is not supported
    world = FakeWorld(dual_camera=True)
    assert type(world.image_lock) is type(SIM_MP_CTX.Semaphore(value=0))
    assert type(world.exit_event) is type(SIM_MP_CTX.Event())

    bridge = FakeBridge(dual_camera=False, high_quality=False)
    assert type(bridge.started) is type(SIM_MP_CTX.Value('i', False))

  def test_bridge_survives_spawning(self):
    # the bridge instance is the spawned process' target, so everything it carries has to be
    # picklable and every shared object has to be inheritable across the spawn
    bridge = FakeBridge(dual_camera=True, high_quality=False)
    bridge.started.value = True
    world = FakeWorld(dual_camera=True)
    world.image_lock.release()

    q = SIM_MP_CTX.Queue()
    p = SIM_MP_CTX.Process(target=_report_state, args=(bridge, world, q))
    p.start()
    try:
      state = q.get(timeout=120)
    finally:
      p.join(30)
      p.kill()

    assert p.exitcode == 0
    assert state == {"cls": "FakeBridge", "dual_camera": True, "high_quality": False, "started": True,
                     "has_params": True, "world": None, "image_lock": True}
    assert world.exit_event.is_set()

  def test_rgb_to_nv12_channel_order(self):
    # the camera readback path has to keep openpilot's RGB layout: a red frame must not end up
    # looking like a blue one once camerad converts it
    red = np.zeros((H, W, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros((H, W, 3), dtype=np.uint8)
    blue[:, :, 2] = 255

    red_y = np.frombuffer(rgb_to_nv12(red)[:W*H], dtype=np.uint8)
    blue_y = np.frombuffer(rgb_to_nv12(blue)[:W*H], dtype=np.uint8)

    # BT.601 weighs red about 2.5x heavier than blue
    assert red_y[0] > blue_y[0]
    assert np.all(red_y == red_y[0])
    assert np.all(blue_y == blue_y[0])

  @unittest.skipIf(metadrive_process is None, "metadrive is not installed")
  def test_macos_panda3d_config(self):
    from panda3d.core import ConfigVariableString

    assert metadrive_process is not None
    with mock.patch("sys.platform", "darwin"):
      metadrive_process.apply_panda3d_config()

    # Apple only exposes GL 4.1, and only through a core profile
    assert ConfigVariableString("gl-version").getValue() == "4 1 core"
    assert "pandagl" in ConfigVariableString("load-display").getValue()
